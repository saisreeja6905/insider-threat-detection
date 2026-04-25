# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import spearmanr, kendalltau, chi2_contingency, f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from IPython.display import display
from django.shortcuts import render
import warnings

# Settings
sns.set_theme(style="darkgrid")
pd.set_option('display.max_rows', 10)
warnings.filterwarnings("ignore")

# View Function
def graphs(request):
    df = pd.read_csv(r"media/augmented_dataset1.csv", low_memory=False)
    
    # Overview
    print(f'Data: {df.shape[0]} records, {df.shape[1]} columns.')
    print(f'Data types:\n{df.dtypes.value_counts()}')
    display(df.head())

    # Missing and Duplicates
    print(f'Missing: {", ".join([f"{col} - {df[col].isnull().mean():.0%}" for col in df])}')
    print(f'Duplicates: {", ".join([f"{col} has duplicates" if df[col].duplicated().any() else f"{col} no duplicates" for col in df])}')

    # Convert to category
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Date':
            df[col] = df[col].astype('category')

    df = df.sort_values(by='Year', ascending=True).reset_index(drop=True)
    display(df.head(2), df.describe())

    # Outlier Detection
    num_cols = ['Year', 'Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
    z_outliers = (np.abs(stats.zscore(df[num_cols]) > 3)).any(axis=1)
    iqr_outliers = ((df[num_cols] < df[num_cols].quantile(0.25) - 1.5 * (df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25))) | 
                    (df[num_cols] > df[num_cols].quantile(0.75) + 1.5 * (df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)))).any(axis=1)
    print("Z-score outliers:\n", df[z_outliers], "\nIQR outliers:\n", df[iqr_outliers])

    # Visualizations
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[num_cols])
    plt.title('Box Plot of Cybersecurity Threats Outliers')
    plt.show()

    cat_cols = ['Country', 'Attack Type', 'Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']
    fig, axes = plt.subplots((len(cat_cols) + 1) // 2, 2, figsize=(15, len(cat_cols) * 2))
    for i, (col, ax) in enumerate(zip(cat_cols, axes.flatten())):
        sns.countplot(y=col, data=df, ax=ax)
        ax.set_title(f'Distribution of {col}')
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, (col, ax) in enumerate(zip(num_cols, axes.flatten())):
        sns.histplot(df[col], bins=10, kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    fig.tight_layout()
    plt.show()

    # Skewness and Kurtosis
    print(f"Skewness: {stats.skew(df[num_cols])}, Kurtosis: {stats.kurtosis(df[num_cols]) + 3}")

    # Correlation Analysis
    corr_pairs = [('Year', 'Financial Loss (in Million $)'), ('Number of Affected Users', 'Financial Loss (in Million $)'),
                  ('Financial Loss (in Million $)', 'Incident Resolution Time (in Hours)'), ('Number of Affected Users', 'Year'),
                  ('Number of Affected Users', 'Incident Resolution Time (in Hours)')]
    for x, y in corr_pairs:
        print(f"Correlation {x} vs {y}: {df[x].corr(df[y]):.2f}")

    sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Relationships Between Numerical Variables')
    plt.show()

    # Non-linear correlations
    for method, func in [('Spearman', spearmanr), ('Kendall', kendalltau)]:
        corr, _ = func(df['Year'], df['Financial Loss (in Million $)'])
        print(f"{method} Correlation: {corr:.2f}")

    # Polynomial Regression
    X, y = df[['Year']], df['Financial Loss (in Million $)']
    X_poly = PolynomialFeatures(degree=2).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    print(f"Polynomial Regression MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X_poly), color='red')
    plt.title('Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('Financial Loss (in Million $)')
    plt.show()

    # Chi-Square Tests
    chi_tests = [
        ('Attack Type', 'Country'), ('Attack Type', 'Attack Source'),
        ('Defense Mechanism Used', 'Security Vulnerability Type'), ('Target Industry', 'Security Vulnerability Type')
    ]
    for x, y in chi_tests:
        table = pd.crosstab(df[x], df[y])
        chi2, p, _, _ = chi2_contingency(table)
        print(f"Chi-Square {x} vs {y}: chi2={chi2:.2f}, p-value={p:.2f}")
        sns.heatmap(table, annot=True, cmap='coolwarm')
        plt.title(f'Heatmap of {x} vs {y}')
        plt.show()

    # ANOVA
    losses = [df.loc[df['Attack Type'] == at, 'Financial Loss (in Million $)'] for at in df['Attack Type'].unique()]
    f_stat, p = f_oneway(*losses)
    print(f"ANOVA Attack Type vs Financial Loss: F={f_stat:.2f}, p={p:.4f}")

    # Trend Visuals
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df.groupby(['Year', 'Attack Type'], observed=False).size().reset_index(name='Count'),
                 x='Year', y='Count', hue='Attack Type', marker='o', palette='Set2')
    plt.title('Attack Types Over Years')
    plt.show()

    # Additional Interactive Plots
    px.line(df.groupby('Year', observed=False).size().reset_index(name='Incident Count'), x='Year', y='Incident Count',
            title='Incidents Over Time', markers=True).show(renderer='iframe')

    px.scatter(df.groupby('Country', observed=False)['Financial Loss (in Million $)'].sum().reset_index(),
               x='Country', y='Financial Loss (in Million $)', size='Financial Loss (in Million $)', color='Country',
               title='Financial Loss by Country').show(renderer='iframe')

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.pivot_table(index='Target Industry', columns='Security Vulnerability Type', aggfunc='size', fill_value=0),
                annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Vulnerabilities by Industry')
    plt.show()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df.groupby('Attack Type', observed=False)['Financial Loss (in Million $)'].mean().reset_index(),
                     x='Attack Type', y='Financial Loss (in Million $)', palette='coolwarm')
    for bar in ax.patches:
        if bar.get_height():
            ax.annotate(f'{int(bar.get_height())}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='bottom')
    plt.title('Average Financial Loss by Attack Type')
    plt.xticks(rotation=45)
    plt.show()

    px.scatter(df, x='Number of Affected Users', y='Financial Loss (in Million $)', color='Attack Type',
               size='Financial Loss (in Million $)', title='Users vs Financial Loss').show(renderer='iframe')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Attack Type', y='Incident Resolution Time (in Hours)', palette='Set2')
    plt.title('Resolution Time by Attack Type')
    plt.xticks(rotation=45)
    plt.show()

    # ---- Classification Model Evaluation ----
    categorical_cols = ['Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']
    numeric_cols = ['Financial Loss (in Million $)', 'Number of Affected Users']

    X = df[categorical_cols + numeric_cols]
    y = df['Attack Type'].astype(str)

    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='mean')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    # Convert sparse matrix to dense array
    X_processed = preprocessor.fit_transform(X).toarray()

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    print("\n--- Model Evaluation ---")
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name}: Accuracy={accuracy_score(y_test, y_pred):.2f}, Precision={precision_score(y_test, y_pred, average='macro'):.2f}, Recall={recall_score(y_test, y_pred, average='macro'):.2f}, F1={f1_score(y_test, y_pred, average='macro'):.2f}")

    return render(request, "users/graph.html", {})