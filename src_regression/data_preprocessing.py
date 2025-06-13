

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocessing(X,y):

    num_col=X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_col=X.select_dtypes(include=['object']).columns.tolist()

    # Numerical pipeline
    num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),   # First impute
        ('scaler', StandardScaler())                     # Then scale
    ])
    cat=Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor=ColumnTransformer([
        ('num',num,num_col),
        ('cat',cat,cat_col)
    ])

    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train,y_train,X_test,y_test,preprocessor







