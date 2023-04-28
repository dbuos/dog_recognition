import mlflow
from ignite.metrics import Accuracy, Loss, Average
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.utils import convert_tensor
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from drecg.data.utils import create_dataloader_validation, create_dataloader_train, create_dataloader_test
from drecg.feature_extraction.distributed import define_model_for_tune
from drecg.feature_extraction.utils import VitLaionPreProcess, create_augmentation_transforms
from drecg.utils import seed_everything, create_mlflow_experiment
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.distributed.rpc import init_rpc
import os


def get_devices():
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device(f'cuda:{i}'))
    return devices


def acc_transform_fn(state):
    y, y_pred, _ = state
    y_pred = torch.round(torch.sigmoid(y_pred))
    return y_pred, y


def log_metrics(metrics, m_type, epoch):
    accuracy = metrics["accuracy"]
    loss = metrics["loss"]
    mlflow.log_metric(f"{m_type}_accuracy", accuracy, step=epoch)
    mlflow.log_metric(f"{m_type}_loss", loss, step=epoch)


def output_transform_fn(_x, y, y_pred, loss=torch.tensor(1.0)):
    return y, y_pred, loss


def prepare_batch_fn(batch, device, non_blocking):
    y_device = torch.device('cuda:3')
    (img_a_batch, img_b_batch), label_batch, _paths = batch
    label_batch = label_batch.reshape(-1, 1).to(torch.float32)

    return (
        (convert_tensor(img_a_batch, device=device, non_blocking=non_blocking),
         convert_tensor(img_b_batch, device=device, non_blocking=non_blocking)),
        convert_tensor(label_batch, device=y_device, non_blocking=non_blocking),
    )


def get_model(devices, microbatch_num):
    return define_model_for_tune(devices, microbatch_num)


def train():
    devices = get_devices()
    model = get_model(devices, 2)

    transforms = VitLaionPreProcess()
    train_transform = create_augmentation_transforms(transforms)
    train_dataloader = create_dataloader_train(root='/home/ubuntu/train', transforms=train_transform, batch_size=16)
    test_dataloader = create_dataloader_test(transforms=transforms, batch_size=16)
    validation_dataloader = create_dataloader_validation(root='/home/ubuntu/validation', transforms=transforms,
                                                         batch_size=16)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    p_bar = ProgressBar()

    train_accuracy = Accuracy(output_transform=acc_transform_fn)
    train_loss = Average(output_transform=lambda out: out[2].item())
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        devices[0],
        non_blocking=True,
        output_transform=output_transform_fn,
        prepare_batch=prepare_batch_fn
    )
    train_accuracy.attach(trainer, "accuracy")
    train_loss.attach(trainer, "loss")

    trainer.state_dict_user_keys.append("best_valid_loss")
    trainer.state_dict_user_keys.append("best_weights")
    trainer.state.best_valid_loss = float("inf")

    def create_evaluator():
        val_metrics = {
            "accuracy": Accuracy(output_transform=acc_transform_fn),
            "loss": Loss(loss_fn, output_transform=lambda out: (out[1], out[0]))}
        return create_supervised_evaluator(
            model,
            device=devices[0],
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
        state_dict = engine.state.best_weights
        path_to_save = f"/home/ubuntu/model.pth"
        torch.save(state_dict, path_to_save)

    p_bar.attach(
        trainer,
        metric_names=["accuracy", "loss"],
        state_attributes=["best_valid_loss"]
    )

    print("Starting...")
    seed_everything(42)
    trainer.run(train_dataloader, max_epochs=5)
    print("Finished!")


if __name__ == '__main__':
    create_mlflow_experiment('Vit Distributed Fine Tuning')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    init_rpc('worker', rank=0, world_size=1)
    train()
