from ignite.engine import Events
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, Average
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.utils import convert_tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.integration import PyTorchIgnitePruningHandler

from drecg.data.utils import create_vector_repr_dataloaders
from drecg.models.feat_extraction import DiffFeatureDetectorParamBiDirectional, ProjectionReductionFeatureDetector, \
    DiffFeatureDetectorParam
from drecg.utils import seed_everything
from torch import optim
import torch
import mlflow
from ignite.contrib.handlers.tqdm_logger import ProgressBar


def define_model_for_trial(trial, model_arch='diff_feature_detector'):
    if model_arch == 'diff_feature_detector':
        model = DiffFeatureDetectorParamBiDirectional(num_hidden=trial.suggest_int("num_hidden", 1, 4),
                                                      hidden_units=trial.suggest_int("hidden_units", 16, 256),
                                                      features_dropout=trial.suggest_float("features_dropout", 0.2,
                                                                                           0.7),
                                                      hidden_dropout=trial.suggest_float("hidden_dropout", 0.2, 0.7),
                                                      hidden_act=trial.suggest_categorical("hidden_act", ["relu"]),
                                                      features_dim=trial.suggest_categorical("feature_dim", [256]))
        return model
    elif model_arch == 'attention_hidden_states':
        model = DiffFeatureDetectorParam(num_hidden=trial.suggest_int("num_hidden", 1, 4),
                                         hidden_units=trial.suggest_int("hidden_units", 16, 256),
                                         features_dropout=trial.suggest_float("features_dropout", 0.2, 0.7),
                                         hidden_dropout=trial.suggest_float("hidden_dropout", 0.2, 0.7),
                                         hidden_act=trial.suggest_categorical("hidden_act", ["relu"]),
                                         features_dim=trial.suggest_categorical("feature_dim", [256]))
        return model
    elif model_arch == 'projection_reduction':
        model = ProjectionReductionFeatureDetector(num_hidden=trial.suggest_int("num_hidden", 1, 3),
                                                   hidden_units=trial.suggest_int("hidden_units", 32, 256),
                                                   features_dropout=trial.suggest_float("features_dropout", 0.2,
                                                                                        0.6),
                                                   hidden_dropout=trial.suggest_float("hidden_dropout", 0.2, 0.7),
                                                   hidden_act=trial.suggest_categorical("hidden_act", ["relu"]),
                                                   projection_dim=trial.suggest_int("projection_dim", 128, 768),
                                                   merge_operation=trial.suggest_categorical("merge_operation",
                                                                                             ["subtract", "concat"]),
                                                   features_dim=trial.suggest_categorical("feature_dim", [1280]))
        return model
    else:
        raise NotImplementedError


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
    (img_a_batch, img_b_batch), label_batch = batch
    label_batch = label_batch.reshape(-1, 1).to(torch.float32)

    return (
        (convert_tensor(img_a_batch, device=device, non_blocking=non_blocking),
         convert_tensor(img_b_batch, device=device, non_blocking=non_blocking)),
        convert_tensor(label_batch, device=device, non_blocking=non_blocking),
    )


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss


def log_params(trial):
    for key, value in trial.params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("pre_operation", 'bidirectional_mean')


def promote_model_to_registry(run_id, artifact_name, model_name):
    return mlflow.register_model(
        f"runs:/{run_id}/{artifact_name}",
        model_name
    )


def log_model_in_registry(model):
    mlflow.pytorch.log_model(model, "model")
    scripted_pytorch_model = torch.jit.script(model)
    mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")


def train(trial, max_epochs):
    train_dataloader, validation_dataloader, test_dataloader = \
        create_vector_repr_dataloaders(root_dir=trial.suggest_categorical("features_dir", ["feat_extracted/uform_v1"]),
                                       batch_size=trial.suggest_int("batch_size", 32, 64, step=32), bidirectional=False)

    model = define_model_for_trial(trial, model_arch=trial.suggest_categorical("model_arch", ["diff_feature_detector"]))
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
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
        prepare_batch=prepare_batch_fn,
    )
    p_bar = ProgressBar()

    train_accuracy.attach(trainer, "accuracy")
    train_loss.attach(trainer, "loss")

    p_bar.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        closing_event_name=Events.COMPLETED,
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

    pruning_handler = PyTorchIgnitePruningHandler(trial, "loss", trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    es_handler = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, es_handler)

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

    log_params(trial)
    seed_everything(42)
    trainer.run(train_dataloader, max_epochs=max_epochs)
    return trainer.state.best_valid_loss
