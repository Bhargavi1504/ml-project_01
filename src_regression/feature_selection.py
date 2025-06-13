import numpy as np 
from sklearn.feature_selection import VarianceThreshold
def feature_selection(d):
    # Sub-task: Low Variance
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(d.select_dtypes(include=np.number))
    low_variance_cols = d.select_dtypes(include=np.number).columns[selector.get_support()]
    d = d[low_variance_cols.tolist() + ['Total Amount']]

    # Sub-task: High Correlation Filter
    corr_matrix = d.drop(columns='Total Amount').corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    d.drop(columns=to_drop, inplace=True)

# Sub-task: Chi-Squared Test#
#from sklearn.feature_selection import chi2, SelectKBest
#X_temp = d.drop(columns='Total Amount')
#y_temp = (d['Total Amount'] > d['Total Amount'].median()).astype(int)
#chi2_selector = SelectKBest(chi2, k='all')
#chi2_selector.fit(abs(X_temp), y_temp)

     # Final feature set for modeling
    x = d.drop(columns='Total Amount')
    y = d['Total Amount']

    return x,y

