import matplotlib.pyplot as plt
import seaborn as sns
def data_cleaning(d):

    #missing values
    #d.isnull().sum() #CHECk--it is showing in customer id
    
    print(d.isnull().sum())
    #outliers
    #duplicates
    # Sub-task: Duplicates
    d = d.drop_duplicates().copy()


    #inconsistent data
    d.loc[:, 'Description'] = d['Description'].str.lower().str.strip()

    # Sub-task: Rare Categories
    country_counts = d['Country'].value_counts()
    rare_countries = country_counts[country_counts < 100].index
    d.loc[:, 'Country'] = d['Country'].replace(rare_countries, 'Other')

    
    d = d.drop(columns='Customer ID')
    numeric_cols= d.select_dtypes(include=['int64','float64']).columns.tolist()
    print(d.columns)


    # Sub-task: Outliers (Boxplot + Capping)
    for i in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=d, x=i)
        plt.title(f'Boxplot for {i}')
        plt.show()

    def cap_outliers_iqr(d, column):
        Q1 = d[column].quantile(0.25)
        Q3 = d[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        d.loc[:, column] = d[column].clip(lower_bound, upper_bound).astype(d[column].dtype)


    for col in numeric_cols:
        cap_outliers_iqr(d, col)

    return d
