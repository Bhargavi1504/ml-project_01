import numpy as np 
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def feature_selection(df):
    # ✅ Ensure only one 'Total Amount' column exists
    if isinstance(df.columns, pd.MultiIndex) or df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # Sub-task: Low Variance
    selector = VarianceThreshold(threshold=0.01)
    num_df = df.select_dtypes(include=np.number).drop(columns=['Total Amount'], errors='ignore')
    selector.fit(num_df)
    selected_cols = num_df.columns[selector.get_support()]
    
    df_filtered = df[selected_cols.tolist() + ['Total Amount']]  # Re-add target

    # Sub-task: High Correlation Filter
    corr_matrix = df_filtered.drop(columns='Total Amount').corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    df_filtered = df_filtered.drop(columns=to_drop)

    # ✅ Final output
    X = df_filtered.drop(columns='Total Amount')
    y = df_filtered['Total Amount']  # ✅ 1D Series

    return X, y



