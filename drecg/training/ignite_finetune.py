from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Average
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.utils import convert_tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau


from drecg.data.utils import create_dataloader_train, create_dataloader_test, \
    create_dataloader_validation
from drecg.feature_extraction.utils import UFormV1FeatureExtractor, create_augmentation_transforms, \
    VitLaionFeatureExtractor
from drecg.utils import seed_everything
from torch import optim
import torch
import mlflow
from ignite.contrib.handlers.tqdm_logger import ProgressBar


class CompleteModelToTune(torch.nn.Module):
    def __init__(self, head_model, feature_extractor_model):
        super().__init__()
        self.head = head_model
        self.feature_extractor = feature_extractor_model

    def forward(self, x):
        features_a, features_b = self.feature_extractor(x)
        return self.head((features_a, features_b))


def define_model_for_tune(model_head_name, feat_ext_name, version='1'):
    model_uri = "models:/{}/{}".format(model_head_name, version)
    head_model = mlflow.pytorch.load_model(model_uri)

    if feat_ext_name == 'UForm_V1':
        from drecg.models.uform import get_model
        vit_model = get_model('unum-cloud/uform-vl-english')
        feat_extractor = UFormV1FeatureExtractor(model=vit_model)
    elif feat_ext_name == 'ViT_LAION':
        feat_extractor = VitLaionFeatureExtractor()
    else:
        raise ValueError(f'Model {feat_ext_name} not found')

    return CompleteModelToTune(head_model, feat_extractor)


def output_transform_fn(_x, y, y_pred, loss=torch.tensor(1.0)):
    return y, y_pred, loss


def acc_transform_fn(state):
    y, y_pred, _ = state
    y_pred = torch.round(torch.sigmoid(y_pred))
    return y_pred, y


def log_metrics(metrics, m_type, epoch):
    accuracy = metrics["accuracy"]
    loss = metrics["loss"]
    mlflow.log_metric(f"{m_type}_accuracy", accuracy, step=epoch)
    mlflow.log_metric(f"{m_type}_loss", loss, step=epoch)


def prepare_batch_fn(batch, device, non_blocking):
    (img_a_batch, img_b_batch), label_batch, _paths = batch
    label_batch = label_batch.reshape(-1, 1).to(torch.float32)

    return (
        (convert_tensor(img_a_batch, device=device, non_blocking=non_blocking),
         convert_tensor(img_b_batch, device=device, non_blocking=non_blocking)),
        convert_tensor(label_batch, device=device, non_blocking=non_blocking),
    )


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss


def promote_model_to_registry(run_id, artifact_name, model_name):
    return mlflow.register_model(
        f"runs:/{run_id}/{artifact_name}",
        model_name
    )


def log_model_in_registry(model):
    mlflow.pytorch.log_model(model, "model")
    # scripted_pytorch_model = torch.jit.script(model)
    # mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")


def train(max_epochs, model_head_name='uform-vl-english', feat_ext_name='UForm_V1'):
    model = define_model_for_tune(model_head_name, feat_ext_name)

    train_transform = create_augmentation_transforms(model.feature_extractor.transforms)
    train_dataloader = create_dataloader_train(transforms=train_transform, batch_size=32)
    test_dataloader = create_dataloader_test(transforms=model.feature_extractor.transforms)
    validation_dataloader = create_dataloader_validation(transforms=model.feature_extractor.transforms)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.to(device)

    train_accuracy = Accuracy(output_transform=acc_transform_fn)
    train_loss = Average(output_transform=lambda out: out[2].item())
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking=True,
        output_transform=output_transform_fn,
        prepare_batch=prepare_batch_fn
    )
    p_bar = ProgressBar()

    train_accuracy.attach(trainer, "accuracy")
    train_loss.attach(trainer, "loss")

    p_bar.attach(
        trainer,
        # event_name=Events.ITERATION_COMPLETED,
        # closing_event_name=Events.COMPLETED,
        metric_names=["accuracy", "loss"],
        state_attributes=["best_valid_loss"]
    )

    trainer.state_dict_user_keys.append("best_valid_loss")
    trainer.state_dict_user_keys.append("best_weights")
    trainer.state.best_valid_loss = float("inf")

    def create_evaluator():
        val_metrics = {
            "accuracy": Accuracy(output_transform=acc_transform_fn),
            "loss": Loss(loss_fn, output_transform=lambda out: (out[1], out[0]))}
        return create_supervised_evaluator(
            model,
            device=device,
            output_transform=output_transform_fn,
            prepare_batch=prepare_batch_fn,
            non_blocking=True,
            metrics=val_metrics)

    val_evaluator = create_evaluator()
    test_evaluator = create_evaluator()

    def do_execute_test(epoch):
        test_evaluator.run(test_dataloader)
        metrics = test_evaluator.state.metrics
        log_metrics(metrics, "test", epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def execute_lr_scheduler(engine):
        _, _, loss = engine.state.output
        scheduler.step(loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def execute_validation(engine):
        val_evaluator.run(validation_dataloader)
        metrics = val_evaluator.state.metrics
        log_metrics(metrics, "valid", engine.state.epoch)
        if engine.state.best_valid_loss > metrics["loss"]:
            engine.state.best_valid_loss = metrics["loss"]
            engine.state.best_weights = model.state_dict()
            log_metrics(metrics, "best_valid", engine.state.epoch)
            do_execute_test(engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def report_training_metrics(engine):
        metrics = engine.state.metrics
        log_metrics(metrics, "train", engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def save_model(engine):
        model.load_state_dict(engine.state.best_weights)
        log_model_in_registry(model)

    # log_params(trial)
    mlflow.log_param('model_head_name', model_head_name)
    mlflow.log_param('feat_ext_name', feat_ext_name)
    mlflow.log_param('learning_rate', 1e-5)
    print("Starting training")
    seed_everything(42)
    # return model, device, train_dataloader
    trainer.run(train_dataloader, max_epochs=max_epochs)
    return trainer.state.best_valid_loss
