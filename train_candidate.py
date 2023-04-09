import random
import torch
from drecg.data.utils import create_vector_repr_dataloaders
from drecg.models.feat_extraction import LFullModel, DiffFeatureDetectorParam
from drecg.training.loops import SanityCallback
from drecg.utils import seed_everything
import mlflow
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

EPOCHS = 100


def define_model(params, feature_dim):
    model = DiffFeatureDetectorParam(num_hidden=params["num_hidden"],
                                     hidden_units=params["hidden_units"],
                                     features_dropout=params["features_dropout"],
                                     hidden_dropout=params["hidden_dropout"],
                                     hidden_act=params["hidden_act"],
                                     features_dim=feature_dim)
    return model


def create_mlflow_experiment():
    exp_name = 'Manual Training Jobs'
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        mlflow.create_experiment(
            exp_name,
            tags={"version": "v1", "type": "vit_features"},
        )
    mlflow.set_experiment(exp_name)


def do_train(params, run_name, features_size, root_dir):
    with mlflow.start_run(run_name=run_name):
        train_dataloader, validation_dataloader, test_dataloader = \
            create_vector_repr_dataloaders(root_dir=root_dir, batch_size=params["batch_size"])
        # callbacks = [PyTorchLightningPruningCallback(trial, monitor="valid_loss"), SanityCallback()]
        callbacks = [SanityCallback()]
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
        trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='gpu', logger=tb_logger, callbacks=callbacks, gpus=1)
        save_checkpoint_fn = lambda mm: torch.save(mm.state_dict(), f"checkpoints/{run_name}.ckpt")
        model = define_model(params, features_size)
        lfull_model = LFullModel(model,
                                 optimizer=params["optimizer"],
                                 lr=params["lr"],
                                 save_checkpoint_fn=save_checkpoint_fn,
                                 )

        trainer.fit(lfull_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
        lfull_model.model.load_state_dict(torch.load(f"checkpoints/{run_name}.ckpt"))

        trainer.test(model=lfull_model, dataloaders=test_dataloader)
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("features_size", features_size)
        mlflow.log_param("features_dir", root_dir)
        mlflow.log_param("pre_operation", 'bidirectional_mean')

if __name__ == "__main__":
    create_mlflow_experiment()
    r_dir = 'features_ext_vit_laion'
    parameters = {
        'num_hidden': 1,
        'hidden_units': 114,
        'features_dropout': 0.3522244827722341,
        'hidden_dropout': 0.541110636043878,
        'hidden_act': 'relu',
        'optimizer': 'AdamW',
        'lr': 1e-4,
        'batch_size': 32,
    }
    f_size = 1280

    r_name = f'run_{random.randint(0, 1000000)}'
    seed_everything(42)
    do_train(parameters, r_name, f_size, r_dir)

    # r_name = f'run_{random.randint(0, 1000000)}'
    # seed_everything(42)
    # do_train(parameters, r_name, f_size, r_dir)

