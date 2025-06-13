from src_regression.Data_Ingestion_reg import load_data,data_understanding
from src_regression.EDA_reg import  eda_reg_univariate,eda_reg_bivariate,eda_reg_multivariate
from src_regression.data_cleaning import data_cleaning
from src_regression.Feature_engineering import feature_engineering
from src_regression.feature_selection import feature_selection
from src_regression.data_preprocessing import preprocessing
import pandas as pd 
from src_regression.model import model

df=load_data(r"C:\Users\BHARGAVI\Downloads\advance_ml\online_retail_II.csv")
s=data_understanding(df)
#eda_reg_univariate(df)
#eda_reg_bivariate(df)
#eda_reg_multivariate(df)
k=data_cleaning(df)                 #outliers cleared
q=feature_engineering(k)            #new columns added
X,Y= feature_selection(q)
print("The Xtrain values",X)
X_train,y_train,X_test,y_test=preprocessing(X,Y)

# Ensure y_train is 1D
if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
   print("y_train columns:", y_train.columns)
   y_train = y_train.iloc[:, 0]  # Or choose by name: y_train['target_column_name']

# Ensure y_test is 1D
if isinstance(y_test, pd.DataFrame) and y_test.shape[1] > 1:
    print("y_test columns:", y_test.columns)
    y_test = y_test.iloc[:, 0]  # Or use y_test['TargetColumnName']

f=model(X_train,y_train,X_test,y_test)

import joblib
joblib.dump(f, 'best_model_bagging.pkl')
