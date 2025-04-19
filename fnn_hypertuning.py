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
from fnn import *
import pickle

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
        try:
            os.makedirs(f"hyperparameters/{self.run_name}")
        except FileExistsError:
            pass

    def objective(self, params):
        # write the params for this run to an output file
        with open(f"hyperparameters/{self.run_name}/fnn_params.txt", "a") as results:
            results.write(f"{params}\n")
        # break up the params dictionary
        #dataset = params['dataset']
        input_size = self.dataset['train_input'].shape[1]
        hidden_nodes = [int(params['hidden_nodes']) for i in range(params['n_layers'])]
        print(f'HIDDEN NODES: {hidden_nodes}')
        output_size = self.dataset['train_output'].shape[1]
        num_epochs = int(params['num_epochs'])
        batch_size = int(params['batch_size'])
        learning_rate = params['learning_rate']
        use_dropout = params['use_dropout']
        dropout_prob = params['dropout_prob']

        # get train and test data from dataset
        train_data = TensorDataset(self.dataset['train_input'], self.dataset['train_output'])
        test_data = TensorDataset(self.dataset['test_input'], self.dataset['test_output'])

        # write dataloaders
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

        # define the model
        model = FNN(input_size, hidden_nodes, output_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            model.train()
            for i, (x_train, y_train) in enumerate(train_loader):
                x_train, y_train = x_train.to(self.device).float(), y_train.to(self.device).float()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(self.device).float(), y_test.to(self.device).float()
                output = model(x_test)
                loss = criterion(output, y_test)
                val_loss += loss.item()
        
        avg_loss = val_loss/len(test_loader)
        with open(f"hyperparameters/{self.run_name}/fnn_MSE.txt", "a") as results:
            results.write(f"AVG MSE: {avg_loss}\n")
        return avg_loss


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
    print(f'Best Hyperparameters for FNN: {best}')
    return 

if __name__ == "__main__":
    space = {
        'hidden_nodes': hp.quniform('hidden_nodes', 32, 512, 32), 
        'num_epochs': hp.quniform('num_epochs', 10, 100, 5),  # Discrete values 
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),  # Discrete 
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)), 
        'use_dropout': hp.choice('use_dropout', [True, False]), 
        'dropout_prob': hp.uniform('dropout_prob', 0.0, 0.5),
        'n_layers': hp.choice('n_layers', [1,2,3,4,5])
        } 
   # get the dataset  
    with open(snakemake.input.dataset, 'rb') as f:
        data_dict = pickle.load(f)

    tuner = Tuner(
                    dataset = data_dict, 
                    run_name = snakemake.config['name'], 
                    space = space, 
                    max_evals = 50,
                    seed = 42, 
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    symbolic = True)
    tune_case(tuner)
    # try:
    #     tune_case(tuner)
    # except Exception as e:
    #     print(f"TUNING INTERRUPTED! Error: {e}")