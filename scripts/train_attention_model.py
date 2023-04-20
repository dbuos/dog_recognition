from datetime import datetime
import mlflow
from drecg.training.ignite import train_features_model
from drecg.models.feat_extraction import AttentionBasedBiDirectionalDetector, AttentionConcatBasedDetector
from drecg.utils import create_mlflow_experiment

EXPERIMENT_NAME = 'Attention Based Detector'
TRACKING_URI = "http://127.0.0.1:5000"

h_params = {
    "max_epochs": 16,
    "features_dir": "/home/daniel/data_dogs/vit_features_hdf5",
    "batch_size": 32,
    "lr": 1e-4,
    "optimizer_name": "AdamW",
    "seed": 42,
    "pruning_handler": None,
    "early_stopping_patience": 20
}

model_params = {
    "features_dim": 1664,
    "layer_norm": True,
    "num_hidden": 2,
    "hidden_units": 64,
    "features_dropout": 0.4,
    "hidden_dropout": 0.5,
    "hidden_act": 'relu',
    "attention_heads": 2,
    "attention_dropout": 0.1,
}


def log_params(params):
    for dic in params:
        for key, value in dic.items():
            mlflow.log_param(key, value)


if __name__ == '__main__':
    create_mlflow_experiment(EXPERIMENT_NAME)
    current_datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with mlflow.start_run(run_name=f"run_{current_datetime_str}"):
        log_params([h_params, model_params])
        # mlflow.log_param("model_arch", 'AttentionConcatBasedDetector')
        # model = AttentionBasedBiDirectionalDetector(**model_params)
        model = AttentionConcatBasedDetector(**model_params)
        train_features_model(model, **h_params)
    print("Done!")
