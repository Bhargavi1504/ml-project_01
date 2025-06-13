#2.exploratoey data analysis
#sub-task:univariate analysis
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import pandas as pd 

def eda_reg_univariate(d):
    # Sub-task: Univariate Analysis for All Numeric Features
    numeric_cols = ['Quantity', 'Price', 'Total Amount']
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(d[col], bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    # Sub-task: KDE Plot
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(d[col], fill=True)
        plt.title(f'KDE Plot for {col}')
        plt.show()

    # Sub-task: Violin Plots
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.violinplot(y=d[col])
        plt.title(f'Violin Plot for {col}')
        plt.show()

def eda_reg_bivariate(d):
    #Bivariate anslysis

    #for num
    numeric_cols= d.select_dtypes(include=['int64','float64']).columns.tolist()
    # Drop ID from numeric analysis because it gives widespread
    if 'Customer ID' in numeric_cols:
        numeric_cols.remove('Customer ID')

    for i in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=d[i],y=d['Total Amount'])
        plt.title(f'{i} vs target')
        plt.xlabel(i)
        plt.ylabel('Total Amount')
        plt.ylim(0, 500)
        plt.show()

    #Sub-task: Optimized Categorical Bivariate Analysis

    # Step 1: Select valid categorical columns with <30 unique categories
    categorical_cols = d.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if d[col].nunique() < 30]

    # Step 2: Sample the dataset
    sampled_data = d.sample(n=2000, random_state=42)

    # Step 3: Plot barplots of top 10 categories
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        mean_vals = sampled_data.groupby(col)['Total Amount'].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=mean_vals.index, y=mean_vals.values)
        plt.title(f'Average Total Amount by {col} (Top 10)')
        plt.xlabel(col)
        plt.ylabel('Avg Total Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def eda_reg_multivariate(d):
    
    # Sub-task: Multivariate Analysis
    numeric_cols= d.select_dtypes(include=['int64','float64']).columns.tolist()
    corr = d[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Sub-task: Correlation Insights
    print("\nCorrelation Matrix:\n", corr)
    strong_corr = corr[(corr > 0.5) | (corr < -0.5)]
    print("\nStrong Correlations (> 0.5 or < -0.5):\n", strong_corr.dropna(how='all').dropna(axis=1, how='all'))

    #This will return a number:
    #Close to 0 → approximately symmetrical
    #> 1 or < -1 → highly skewed
    #Between -0.5 and 0.5 → fairly symmetrical

    # Sub-task: Skewness Check and Fix

    skewed_feats = d[numeric_cols].skew().sort_values(ascending=False)
    print("\nSkewness Before Transformation:\n", skewed_feats)

    pt = PowerTransformer()
    d[numeric_cols] = pt.fit_transform(d[numeric_cols])

    skewed_after = pd.DataFrame(d[numeric_cols].skew(), columns=['Skewness_After'])
    print("\nSkewness After Power Transformation:\n", skewed_after)




