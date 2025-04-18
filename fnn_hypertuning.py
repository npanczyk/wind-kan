import numpy as np
import torch
from hyperopt import tpe, hp, fmin, Trials, rand
from preprocessing import *
from functools import partial
from sklearn.metrics import r2_score
from kan import *
from accessories import *
import shutil
import os
import csv
import datetime as dt

torch.set_default_dtype(torch.float64)

########################  TUNER CLASS ###########################
class Tuner():
    """Tuner objects pair objective functions to a dataset to feed into hyperopt tuning procedure.
    """
    def __init__(self, dataset, run_name, space, max_evals, seed, device, symbolic=False):
        self.dataset = dataset
        self.run_name = run_name
        self.space = space
        self.max_evals = max_evals
        self.seed = seed
        self.device = device
        self.symbolic = symbolic
        # make sure our file structure is ready to go
        try:
            os.makedirs(f"hyperparameters/{self.run_name}")
        except FileExistsError:
            files = os.listdir(f"hyperparameters/{self.run_name}")
            for file in files:
                os.remove(f"hyperparameters/{self.run_name}/{file}")

    def objective(self, params):
        # write the params for this run to an output file
        with open(f"hyperparameters/{self.run_name}/kan_params.txt", "a") as results:
            results.write(f"{params}\n")
        # break up the params dictionary
        dataset = params['dataset'](cuda=True)
        input_size = dataset['train_input'].shape[1]
        hidden_nodes = params['hidden_nodes']
        output_size = dataset['train_output'].shape[1]
        num_epochs = params['num_epochs']
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        use_dropout = params['use_dropout']
        dropout_prob = params['dropout_prob']

        """FINISH OBJECTIVE FUNC HERE
        """


def tune(obj, space, max_evals, algorithm=None):
    trials = Trials()
    best = fmin(obj, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best, trials

def tune_case(tuner):
    """Runs the hyperopt tuner and saves the best results for that case under the /hyperparameters directory. 

    Args:
        tuner (Tuner object): Tuner object defined by a specific dataset and hyperparameter space dictionary.
    """
    best, trials = tune(
                    obj=tuner.objective, 
                    space=tuner.space, 
                    max_evals=tuner.max_evals)
    return 

if __name__ == "__main__":
    space = {
        "depth": hp.choice("depth", [1, 2]),
        "grid": hp.choice("grid", [3, 4, 5, 6, 7, 8, 9, 10]),
        "k": hp.choice("k", [2, 3, 4, 5, 6, 7, 8]),
        "steps": hp.choice("steps", [25, 50, 75, 100, 125, 150, 200, 250]),
        "lamb": hp.uniform("lamb", 0, 0.001),
        "lamb_entropy": hp.uniform("lamb_entropy", 0, 10),
        "lr_1": hp.choice("lr_1", [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]), 
        "lr_2": hp.choice("lr_2", [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]),
        "reg_metric": hp.choice("reg_metric", ["edge_forward_spline_n",
                                                "edge_forward_sum",
                                                "edge_forward_spline_u",
                                                "edge_backward",
                                                "node_backward"])
            }
    tuner = Tuner(
                    dataset = snakemake.input.dataset, 
                    run_name = snakemake.input.run_name, 
                    space = space, 
                    max_evals = 1,
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    symbolic = True)
    try:
        tune_case(tuner)
    except Exception as e:
        print(f"TUNING INTERRUPTED! Error: {e}")