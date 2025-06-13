from src_regression.Data_Ingestion_reg import load_data, data_understanding
from src_regression.EDA_reg import eda_reg_univariate, eda_reg_bivariate, eda_reg_multivariate
from src_regression.data_cleaning import data_cleaning
from src_regression.Feature_engineering import feature_engineering
from src_regression.feature_selection import feature_selection
from src_regression.data_preprocessing import preprocessing
from src_regression.model import model
import pandas as pd
import joblib

# 1️⃣ Load and Understand Data
df = load_data(r"C:\Users\BHARGAVI\Downloads\advance_ml\online_retail_II.csv")
s = data_understanding(df)

# 2️⃣ Data Cleaning and Feature Engineering
cleaned_df = data_cleaning(df)
engineered_df = feature_engineering(cleaned_df)

# 3️⃣ Feature Selection
X, y = feature_selection(engineered_df)

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)
print("✅ y type:", type(y))
print("✅ y name:", y.name)

# 4️⃣ Preprocessing
X_train, y_train, X_test, y_test ,preprocessor= preprocessing(X, y)

# 5️⃣ Ensure y is 1D Series
if isinstance(y_train, pd.DataFrame):
    if y_train.shape[1] > 1:
        print("ℹ️ Fixing y_train shape")
        y_train = y_train.iloc[:, 0]
    else:
        y_train = y_train.squeeze()
else:
    y_train = pd.Series(y_train)

if isinstance(y_test, pd.DataFrame):
    if y_test.shape[1] > 1:
        print("ℹ️ Fixing y_test shape")
        y_test = y_test.iloc[:, 0]
    else:
        y_test = y_test.squeeze()
else:
    y_test = pd.Series(y_test)

print("✅ Final y_train shape:", y_train.shape)
print("✅ Final y_test shape:", y_test.shape)

# 6️⃣ Train the model
best_model = model(X_train, y_train, X_test, y_test)

from sklearn.pipeline import Pipeline

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', best_model)
])

full_pipeline.fit(X, y)

# 7️⃣ Save the model
joblib.dump(full_pipeline, 'final_model_pipeline.pkl')


