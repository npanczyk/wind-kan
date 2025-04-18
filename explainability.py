from preprocessing import *
from accessories import print_shap
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sympy import symbols, sympify, lambdify
from functools import partial
import datetime as dt

def kan_shap(equation_file, X_train, X_test, input_names, output_names, save_as, k, width=0.2):
    """Uses kernel SHAP to generate feature importances for each input feature and for each output using symbolic equation generated by KAN for each output. 

    Args:
        equation_file (str): .txt file containing a sympy str expression for each output in a dataset, where each output is separated by a newline
        X_test (NumPy array): feature values for the test set
        Y_test (NumPy array): output values for the test set
        input_names (list): labels for features
        output_names (list): labels for outputs
        save_as (str): name to save the file
        shap_range (int, optional): number of rows used by SHAP's Kernel Explainer. See SHAP documentation for more detail. Defaults to 300.
        width (float, optional): width of bars on output plot. Defaults to 0.2.

    Returns:
        tuple: fig, ax
    """
    shap_mean_list = []
    n = len(output_names)
    with open(equation_file) as file:
        exprs = [file.readline() for line in range(n)]
    for output, expr in zip(output_names, exprs):
        num_vars = len(input_names)
        # convert string into a sympy expression object
        s_expr = sympify(expr)
        # equation as function with arg "inputs"
        variables = symbols('x_1:%d' % (num_vars + 1))
        print(variables)
        compute_Y = lambdify([variables], s_expr)
        # wrapper function, called model
        model = lambda inputs: np.array([compute_Y(variables) for variables in inputs])
        X_train_summary = shap.kmeans(X_train, k)
        explainer = shap.KernelExplainer(model, X_train_summary)
        shap_values = explainer.shap_values(X_test[0:])
        # remove nan values from array (this is only necessary for MITR-A)
        # equation has a maybe negative number raised to a fractional power
        # this throws an error, so we drop these cases
        shap_values = shap_values[~np.isnan(shap_values).any(axis=1)]
        shap_mean_list.append(pd.DataFrame(np.abs(shap_values).mean(axis=0),columns=[output],index=input_names))
    shap_mean = pd.concat(shap_mean_list, axis=1)
    print(shap_mean)
    if not os.path.exists('shap-values'):
        os.makedirs('shap-values')
    path = f'shap-values/{save_as}_kan_{str(dt.date.today())}.pkl'
    shap_mean.to_pickle(path)
    return path

def get_kan_shap(datasets_dict):
    shap_paths = {}
    for model, info in datasets_dict.items():
        dataset = info[0](cuda=False)
        X_test = dataset['test_input'].detach().numpy()
        X_train = dataset['train_input'].detach().numpy()
        input_names = dataset['feature_labels']
        output_names = dataset['output_labels']
        equation_file = info[1]
        train_samples = X_train.shape[0]
        input_size = X_train.shape[1]
        k = int(np.round(0.005*train_samples*input_size))
        if k > 100:
            k = 100
        save_as = f"{model.upper()}"
        path = kan_shap(equation_file, X_train, X_test, input_names, output_names, save_as, k=k, width=0.2)
        shap_paths[model] = path
    return shap_paths

def fnn_shap(model, X_train, X_test, input_names, output_names, save_as, k=50, kcheck=False):
    """gets feature importances using kernel shap for an fnn

    Args:
        model (pytorch model obj): _description_
        X_train (numpy array): 
        X_test (numpy array): _description_
        input_names (_type_): _description_
        output_names (_type_): _description_
        save_as (_type_): _description_
        k (int, optional): Number of means used for shap.kmeans approximation of training data. Defaults to 50.

    Returns:
        _type_: _description_
    """
    model_pred = lambda inputs: model(torch.tensor(inputs, dtype=torch.float32)).cpu().detach().numpy()
    X_train_summary = shap.kmeans(X_train, k)
    explainer = shap.KernelExplainer(model_pred, X_train_summary)
    shap_values = explainer.shap_values(X_test[0:])
    shap_mean = pd.DataFrame(np.abs(shap_values).mean(axis=0),columns=[output_names],index=input_names)
    if not os.path.exists('shap-values'):
        os.makedirs('shap-values')
    path = f'shap-values/{save_as}_fnn_{str(dt.date.today())}.pkl'
    shap_mean.to_pickle(path)
    if kcheck:
        fig, ax = plt.subplots()
        ax.scatter(X_train[:,0], X_train[:,1], label='X_train')
        ax.scatter(X_train_summary.data[:,0], X_train_summary.data[:,1], label='X_train_summary', color='red')
        ax.legend()
        ax.set_xlabel(f'{input_names[0]}')
        ax.set_ylabel(f'{input_names[1]}')
        ax.set_title(f'k = {k}')
        plt.savefig(f'figures/fnn-shap/{save_as}_KCHECK.png', dpi=300)
    return path

def plot_shap(path, save_as, type='kan', width=0.2):
    """Makes bar plots for shap values of a given dataset.

    Args:
        path (str): path to shap values file
        save_as (str): name of run
        type (str, optional): Model type, either fnn or kan. Defaults to 'kan'.
        width (float, optional): Width of plotted bars. Defaults to 0.2.

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    plt.rcParams.update({'font.size': 16})
    shap_mean = pd.read_pickle(path)
    fig, ax = plt.subplots(figsize=(10,6))
    x_positions = np.arange(len(shap_mean.index))
    output_names = list(shap_mean.columns)
    input_names = list(shap_mean.index)
    for i, col in enumerate(shap_mean.columns):
        # very stupid label thing but it has to do with a list of tuples for fnn
        if type.upper()=='FNN':
            label = output_names[i][0]
        else:
            label = output_names[i]
        ax.bar(x_positions + i*width, shap_mean[col], capsize=4, width=width, label=label)
    ax.set_ylabel("Mean of |SHAP Values|")
    ax.set_yscale('log')
    fig.legend(title='Output', loc='outside upper center', ncol=5)
    n = len(output_names)
    ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    ax.set_xticks(x_positions + (n-1)*width/2)
    ax.set_xticklabels(input_names, rotation=50)
    plt.subplots_adjust(bottom=0.27, left=0.15)
    #plt.tight_layout()
    if not os.path.exists('figures/shap'):
        os.makedirs('figures/shap')
    plt.savefig(f'figures/shap/{save_as}.png', dpi=300)
    return fig, ax

def plot_stacked(kan_path, fnn_path, save_as, width=0.2):
    """Makes bar plots for shap values of KAN and FNN for one dataset stacked.

    Args:
        kan_path (str): path to shap values file for kan
        fnn_path (str): path to shap values file for fnn
        save_as (str): name of run
        type (str, optional): Model type, either fnn or kan. Defaults to 'kan'.
        width (float, optional): Width of plotted bars. Defaults to 0.2.

    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    plt.rcParams.update({'font.size': 16})
    kan_shap_mean = pd.read_pickle(kan_path)
    fnn_shap_mean = pd.read_pickle(fnn_path)
    fig, ax = plt.subplots(figsize=(10,6))
    x_positions = np.arange(len(kan_shap_mean.index))
    output_names = list(kan_shap_mean.columns)
    input_names = list(kan_shap_mean.index)
    for i, col in enumerate(kan_shap_mean.columns):
        label = f'{output_names[i]} KAN'
        ax.bar(x_positions + i*width, kan_shap_mean[col], capsize=4, width=width, label=label)
    for i, col in enumerate(fnn_shap_mean.columns):
        label = f'{output_names[i]} FNN'
        ax.bar(x_positions + i*width + width, fnn_shap_mean[col], capsize=4, width=width, label=label)
    ax.set_ylabel("Mean of |SHAP Values|")
    ax.set_yscale('log')
    ax.legend(title='Output', ncols=len(output_names))
    n = len(output_names)
    ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    ax.set_xticks(x_positions + (n-1)*width/2)
    ax.set_xticklabels(input_names, rotation=45)
    plt.tight_layout()
    if not os.path.exists('figures/shap'):
        os.makedirs('figures/shap')
    plt.savefig(f'figures/shap/{save_as}_stacked.png', dpi=300)
    return fig, ax

if __name__=="__main__":
    datasets_dict = {
        'fp': [get_fp, 'equations/FP_2025-03-04.txt'],
        # 'bwr': [get_bwr, 'equations/BWR_2025-03-05.txt'],
        # 'heat': [get_heat, 'equations/HEAT_2025-03-05.txt'],
        'htgr': [get_htgr, 'equations/HTGR_2025-03-05.txt'],
        # 'mitr_a': [partial(get_mitr, region='A'), 'equations/MITR_A_2025-03-05.txt'],
        # 'mitr_b': [partial(get_mitr, region='B'), 'equations/MITR_B_2025-03-05.txt'],
        # 'mitr_c': [partial(get_mitr, region='C'), 'equations/MITR_C_2025-03-05.txt'],
        # 'mitr': [partial(get_mitr, region='FULL'), 'equations/MITR_2025-03-05.txt'],
        # 'chf': [get_chf, 'equations/CHF_2025-03-05.txt'],
        # 'rea': [get_rea, 'equations/REA_2025-03-05.txt'],
        'xs': [get_xs, 'equations/XS_2025-03-05.txt']
    }

    shap_path_dict = {
        # 'fp': 'shap-values/FP_kan_2025-04-03.pkl', 
        # 'bwr': 'shap-values/BWR_kan_2025-03-18.pkl', 
        # 'heat': 'shap-values/HEAT_kan_2025-03-18.pkl', 
        # 'htgr': 'shap-values/HTGR_kan_2025-04-03.pkl', 
        # 'mitr_a': 'shap-values/MITR_A_kan_2025-03-18.pkl', 
        # 'mitr_b': 'shap-values/MITR_B_kan_2025-03-18.pkl', 
        'mitr_c': 'shap-values/MITR_C_kan_2025-03-18.pkl', 
        # 'mitr': 'shap-values/MITR_kan_2025-03-18.pkl', 
        # 'chf': 'shap-values/CHF_kan_2025-03-18.pkl', 
        # 'rea': 'shap-values/REA_kan_2025-03-18.pkl', 
        # 'xs': 'shap-values/XS_kan_2025-04-03.pkl'
        }

    # # uncomment to calculate kan shap values
    # paths_dict = get_kan_shap(datasets_dict)
    # print(paths_dict)

    # uncomment to make shap kan plots
    for model, path in shap_path_dict.items():
        plot_shap(path, save_as=f'{model}_kan', type='kan', width=0.05)

    ## uncomment to print shap values
    # for model, path in shap_path_dict.items():
    #     print_shap(path, save_as=f'{model}', type='kan')

    # # make stacked plot for CHF and HEAT
    # plot_stacked(kan_path=shap_path_dict['chf'], fnn_path='shap-values/CHF_fnn_2025-03-18.pkl', save_as='CHF', width=0.2)
