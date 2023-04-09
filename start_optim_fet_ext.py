import optuna
import torch
from optuna.trial import TrialState
from drecg.data.utils import create_vector_repr_dataloaders
from drecg.models.feat_extraction import LFullModel, DiffFeatureDetectorParamBiDirectional
from drecg.training.loops import SanityCallback
import mlflow
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from drecg.utils import seed_everything

EPOCHS = 100


def define_model(trial, feature_dim):
    model = DiffFeatureDetectorParamBiDirectional(num_hidden=trial.suggest_int("num_hidden", 1, 3),
                                                  hidden_units=trial.suggest_int("hidden_units", 40, 150),
                                                  features_dropout=trial.suggest_float("features_dropout", 0.2, 0.6),
                                                  hidden_dropout=trial.suggest_float("hidden_dropout", 0.3, 0.7),
                                                  hidden_act=trial.suggest_categorical("hidden_act", ["relu"]),
                                                  features_dim=feature_dim)
    return model


def create_objective(root_dir='features_ext_vit', features_size=1024):
    def objective(trial: optuna.trial.Trial) -> float:
        with mlflow.start_run(run_name=f"run_{trial.number}"):
            train_dataloader, validation_dataloader, test_dataloader = \
                create_vector_repr_dataloaders(root_dir=root_dir,
                                               batch_size=trial.suggest_int("batch_size", 32, 64, step=32))
            # callbacks = [PyTorchLightningPruningCallback(trial, monitor="valid_loss"), SanityCallback()]
            callbacks = [SanityCallback()]
            tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
            trainer = pl.Trainer(max_epochs=EPOCHS, accelerator='gpu', logger=tb_logger, callbacks=callbacks, gpus=1)
            save_checkpoint_fn = lambda mm: torch.save(mm.state_dict(), f"checkpoints/{trial.number}.ckpt")
            model = define_model(trial, features_size)
            lfull_model = LFullModel(model,
                                     optimizer=trial.suggest_categorical("optimizer", ["AdamW"]),
                                     lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
                                     save_checkpoint_fn=save_checkpoint_fn,
                                     )

            seed_everything(42)
            trainer.fit(lfull_model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
            lfull_model.model.load_state_dict(torch.load(f"checkpoints/{trial.number}.ckpt"))

            trainer.test(model=lfull_model, dataloaders=test_dataloader)
            for key, value in trial.params.items():
                mlflow.log_param(key, value)
            mlflow.log_param("features_size", features_size)
            mlflow.log_param("features_dir", root_dir)
            mlflow.log_param("pre_operation", 'bidirectional_mean')
        return lfull_model.best_valid_loss

    return objective


def create_mlflow_experiment():
    exp_name = 'Feature Extraction + Optuna Test'
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        mlflow.create_experiment(
            exp_name,
            tags={"version": "v1", "type": "vit_features_concat_dense"},
        )
    mlflow.set_experiment(exp_name)


if __name__ == "__main__":
    create_mlflow_experiment()
    # pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    opt_objective = create_objective(root_dir='features_ext_vit_laion', features_size=1280)
    study.optimize(opt_objective, n_trials=2)
    print(study.best_params)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print('Pruned:', len(pruned_trials))
    print(pruned_trials)
    print('Complete trials: ', len(complete_trials))
    print(complete_trials)
