from ignite.metrics import Accuracy, Loss
from ignite.engine import create_supervised_evaluator
from ignite.utils import convert_tensor
from drecg.data.utils import create_dataloader_validation
from drecg.feature_extraction.distributed import define_model_for_tune
from drecg.feature_extraction.utils import VitLaionPreProcess
from drecg.utils import seed_everything
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


def output_transform_fn(_x, y, y_pred, loss=torch.tensor(1.0)):
    return y, y_pred, loss


def prepare_batch_fn(batch, device, non_blocking):
    (img_a_batch, img_b_batch), label_batch, _paths = batch
    label_batch = label_batch.reshape(-1, 1).to(torch.float32)

    return (
        (convert_tensor(img_a_batch, device=device, non_blocking=non_blocking),
         convert_tensor(img_b_batch, device=device, non_blocking=non_blocking)),
        convert_tensor(label_batch, device=device, non_blocking=non_blocking),
    )


def get_model(devices, microbatch_num):
    return define_model_for_tune(devices, microbatch_num)


def run_eval():
    devices = get_devices()
    model = get_model(devices, 2)

    transforms = VitLaionPreProcess()
    # train_dataloader = create_dataloader_train(transforms=transforms, batch_size=16)
    # test_dataloader = create_dataloader_test(transforms=transforms, batch_size=16)
    validation_dataloader = create_dataloader_validation(root='/home/ubuntu/validation', transforms=transforms, batch_size=16)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    p_bar = ProgressBar()

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

    p_bar.attach(
        val_evaluator,
        metric_names=["accuracy", "loss"]
    )

    print("Starting...")
    seed_everything(42)

    val_evaluator.run(validation_dataloader)
    metrics = val_evaluator.state.metrics
    print(f"Results - Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    init_rpc('worker', rank=0, world_size=1)
    run_eval()
