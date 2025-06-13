
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor,BaggingRegressor,StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def model(X_train,y_train,X_test,y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, max_features='sqrt', n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=20, random_state=42),
        'Bagging': BaggingRegressor(n_estimators=20, n_jobs=-1, random_state=42),
        #'SVR': SVR(),
        'KNN': KNeighborsRegressor(algorithm='ball_tree', n_neighbors=3)

    }

    print("\nModel Comparison:")


    scoring = ['r2', 'neg_mean_absolute_error']
    model_scores = {}

    print("\nModel Comparison:")
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
        r2 = scores['test_r2'].mean()
        mae = -scores['test_neg_mean_absolute_error'].mean()  # convert back to positive
        model_scores[name] = {'R2': r2, 'MAE': mae}
        print(f"{name} => R2: {r2:.4f}, MAE: {mae:.4f}")

    best_model_name = max(model_scores, key=lambda k: model_scores[k]['R2'])
    print("\nBest Model based on R2 Score:", best_model_name)


    # Ensure target is 1D (important for sklearn regressors)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Fit the best model (BaggingRegressor in your case)
    best_model = models[best_model_name]  # best_model_name should be 'Bagging'
    best_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # 8️⃣ Evaluation Metrics
    print("\n📊 Evaluation Metrics:")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE  : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R2   : {r2_score(y_test, y_pred):.4f}")

    # 📉 Error Analysis: Residual distribution
    error = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(error, kde=True, color='royalblue', bins=30)
    plt.title("Prediction Error Distribution", fontsize=14)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=np.ravel(y_test), y=np.ravel(y_pred), color='green', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='red', linestyle='--')
    plt.title("Actual vs Predicted", fontsize=14)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return best_model_name
