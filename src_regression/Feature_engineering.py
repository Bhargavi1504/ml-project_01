import pandas as pd 
def feature_engineering(d):
    # Sub-task: Date Feature Extraction
    d['InvoiceDate'] = pd.to_datetime(d['InvoiceDate'])  # Added error handling
    d['Hour'] = d['InvoiceDate'].dt.hour
    d['Weekday'] = d['InvoiceDate'].dt.weekday

    # Sub-task: Binning
    d['PriceBin'] = pd.qcut(d['Price'],q=4,labels=False)

    ## Sub-task: Text Features (Length of description)
    d['Desclength'] = d['Description'].apply(lambda x: len(str(x)))

    # Sub-task: Polynomial Features
    #from sklearn.preprocessing import PolynomialFeatures
    #poly=PolynomialFeatures(degree=2,include_bias=False)
    #poly_feats = poly.fit_transform(d[['Quantity', 'Price']])
    #poly_cols = poly.get_feature_names_out(['Quantity', 'Price'])
    #poly_df = pd.DataFrame(poly_feats, columns=poly_cols)
    #d = pd.concat([d.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

    # Sub-task: Aggregation Features
    #agg_d = d.groupby('Customer ID').agg({
        #'TotalAmount': ['sum', 'mean'],
        #'Quantity': ['sum', 'mean']
    #}).reset_index()
    #agg_d.columns = ['Customer ID', 'TotalSpent', 'AvgSpent', 'TotalQty', 'AvgQty']
    #df = d.merge(agg_d, on='Customer ID', how='left')

    # Remove duplicate columns by keeping the first occurrence
    d = d.loc[:, ~d.columns.duplicated()]

    # Sub-task: Domain Features (e.g., return status)
    d['IsReturn'] = d['Invoice'].astype(str).str.startswith('C').astype(int)

    print(d.columns)

    return d


