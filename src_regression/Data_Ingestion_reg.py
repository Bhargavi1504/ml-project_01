#0.project initialization
#load dataset
import pandas as pd 
def load_data(path):
    d=pd.read_csv(path)
    return d

def data_understanding(d):
    #1.data understanding
    d.shape       #shape of dataset
    d.dtypes      #datatypes
    d.nunique()   #unique values 
    d.columns 
    #here i dont find the target variable so iam creating new variable
    d['Total Amount'] = d['Quantity'] * d['Price']
    #d.value_counts()       #target variable for classification task
    d['Total Amount'].describe()

    #checking the values
    print('Negative values:',(d['Total Amount']<0).sum())
    print("Zero values:",(d['Total Amount']==0).sum())

    #removing negative values
    d=d[d['Total Amount'] >0 ]

    print("negative values after removal:",(d['Total Amount']<0).sum())
    print("Zero values after removal:",(d['Total Amount']==0).sum())
    
    #target distribution
    d['Total Amount'].describe()

    return d