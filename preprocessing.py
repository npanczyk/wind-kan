import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
import torch
import pickle

def get_month(file, month=1):
    """Filters a specific month out of the dataset.

    Args:
        file (str, optional): Filepath for csv containing dataset
        month (int, optional): Numerical month of interest. Defaults to 1 (January).

    Returns:
        Pandas DataFrame: a filtered version of the whole dataset (by month of interest)
    """
    df = pd.read_csv(file)
    month_cols = []
    for col in df.columns:
        if '_' not in col:
            month_cols.append(col)
        elif col.split('_')[-1] != str(month):
            pass
        else:
            month_cols.append(col)
    return df[month_cols], month

def get_wind(file='usa_0_regional_monthly.csv', cuda=True):
    df, month = get_month(file)
    # get names of features by month
    feature_cols = ['Year', 'lat', 'lng', 'alt', f'T2M_{month}',
       f'T2M_MAX_{month}', f'T2M_MIN_{month}', f'PS_{month}', f'QV2M_{month}',
       f'PRECTOTCORR_{month}', f'ALLSKY_SFC_SW_DWN_{month}', f'EVPTRNS_{month}', f'GWETPROF_{month}',
       f'SNODP_{month}', f'T2MDEW_{month}', f'CLOUD_AMT_{month}', f'EVLAND_{month}', f'T2MWET_{month}', f'FRSNO_{month}',
       f'ALLSKY_SFC_LW_DWN_{month}', f'ALLSKY_SFC_PAR_TOT_{month}', f'ALLSKY_SRF_ALB_{month}',
       f'PW_{month}', f'Z0M_{month}', f'RHOA_{month}', f'RH2M_{month}', f'CDD18_3_{month}', f'HDD18_3_{month}', f'TO3_{month}',
       f'AOD_55_{month}', f'VAP_{month}', f'VPD_{month}', f'ET0_{month}']
    # get names of output by month
    output_cols = [f'WD2M_{month}', f'WS2M_{month}']
    # get numerical feature columns
    feature_icols = [i for i in range(2,9)] + [i for i in range(11,37)]
    # create a new df for features only
    features_df = df.iloc[:, feature_icols]
    # create a new df for outputs only
    outputs_df = df.iloc[:, [9,10]]
    # split into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
    features_df, outputs_df, test_size=0.3, random_state=42)
    # set device for pytorch
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    # define the minmax scaler
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()
    # scale all train and test sets
    X_train = scaler_X.fit_transform(x_train)
    X_test = scaler_X.transform(x_test)
    Y_train = scaler_Y.fit_transform(y_train)
    Y_test = scaler_Y.transform(y_test)
    # convert to tensors
    train_input = torch.tensor(X_train, dtype=torch.double).to(device)
    train_output = torch.tensor(Y_train, dtype=torch.double).to(device)
    test_input = torch.tensor(X_test, dtype=torch.double).to(device)
    test_output = torch.tensor(Y_test, dtype=torch.double).to(device)
    # create the dataset dictionary
    dataset = {
        'train_input': train_input,
        'train_output': train_output,
        'test_input': test_input,
        'test_output': test_output,
        'feature_labels': feature_cols,
        'output_labels': output_cols,
        'y_scaler': scaler_Y
    }
    name = ''.join(file.split('_')[:2])
    with open(f'{name}_dict.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    return

if __name__=="__main__":
    get_wind(snakemake.input.file, cuda=True)