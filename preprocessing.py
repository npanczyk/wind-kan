import pandas as pd
import numpy as np

def get_month(month=1):
    """Filters a specific month out of the dataset.

    Args:
        month (int, optional): Numerical month of interest. Defaults to 1 (January).

    Returns:
        Pandas DataFrame: a filtered version of the whole dataset (by month of interest)
    """
    df = pd.read_csv('usa_0_regional_monthly.csv')
    month_cols = []
    for col in df.columns:
        if '_' not in col:
            month_cols.append(col)
        elif col.split('_')[-1] != str(month):
            pass
        else:
            month_cols.append(col)
    return df[month_cols]

get_month()