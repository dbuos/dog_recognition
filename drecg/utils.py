import os
import random

import mlflow
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed_everything done: ', seed)


def create_mlflow_experiment(exp_name, tags=None):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        if not tags:
            tags = {"version": "v1"}
        mlflow.create_experiment(
            exp_name,
            tags=tags,
        )
    mlflow.set_experiment(exp_name)