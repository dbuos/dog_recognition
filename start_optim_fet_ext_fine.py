import optuna
from optuna.trial import TrialState
import mlflow
from drecg.training.ignite import train as train_ignite

EPOCHS = 100


def create_objective():
    def objective(trial: optuna.trial.Trial) -> float:
        with mlflow.start_run(run_name=f"run_0_{trial.number}"):
            best_valid_loss = train_ignite(trial, EPOCHS)
        return best_valid_loss

    return objective


def create_mlflow_experiment():
    exp_name = 'Uform Features'
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        mlflow.create_experiment(
            exp_name,
            tags={"version": "v1", "type": "uform_features_diff"},
        )
    mlflow.set_experiment(exp_name)


if __name__ == "__main__":
    create_mlflow_experiment()
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    opt_objective = create_objective()
    study.optimize(opt_objective, n_trials=250)
    print(study.best_params)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print('Pruned:', len(pruned_trials))
    print(pruned_trials)
    print('Complete trials: ', len(complete_trials))
    print(complete_trials)