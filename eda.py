# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from django.shortcuts import render
from django.conf import settings
import os
import warnings
import base64
from io import BytesIO

# Settings
sns.set_theme(style="darkgrid")
pd.set_option('display.max_rows', 10)
warnings.filterwarnings("ignore")


def save_plot_to_base64():
    """Helper function to convert matplotlib plot to base64 string"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode()


def graphs(request):
    df = pd.read_csv(r"media/augmented_dataset1.csv", low_memory=False)
    
    # Store all plot images
    plots = {}
    
    # Overview
    print(f'Data: {df.shape[0]} records, {df.shape[1]} columns.')
    print(f'Data types:\n{df.dtypes.value_counts()}')
    print(df.head())

    # Missing and Duplicates
    print(f'Missing: {", ".join([f"{col} - {df[col].isnull().mean():.0%}" for col in df])}')
    print(f'Duplicates: {", ".join([f"{col} has duplicates" if df[col].duplicated().any() else f"{col} no duplicates" for col in df])}')

    # Convert to category
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Date':
            df[col] = df[col].astype('category')

    df = df.sort_values(by='Year', ascending=True).reset_index(drop=True)
    print(df.head(2))
    print(df.describe())

    # Outlier Detection
    num_cols = ['Year', 'Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
    z_outliers = (np.abs(stats.zscore(df[num_cols]) > 3)).any(axis=1)
    iqr_outliers = ((df[num_cols] < df[num_cols].quantile(0.25) - 1.5 * (df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25))) | 
                    (df[num_cols] > df[num_cols].quantile(0.75) + 1.5 * (df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)))).any(axis=1)
    print("Z-score outliers:\n", df[z_outliers], "\nIQR outliers:\n", df[iqr_outliers])

    # Visualizations - Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[num_cols])
    plt.title('Box Plot of Cybersecurity Threats Outliers')
    plots['boxplot'] = save_plot_to_base64()

    # Categorical Distribution
    cat_cols = ['Country', 'Attack Type', 'Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used']
    fig, axes = plt.subplots((len(cat_cols) + 1) // 2, 2, figsize=(15, len(cat_cols) * 2))
    for i, (col, ax) in enumerate(zip(cat_cols, axes.flatten())):
        sns.countplot(y=col, data=df, ax=ax)
        ax.set_title(f'Distribution of {col}')
    fig.tight_layout()
    plots['categorical_dist'] = save_plot_to_base64()

    # Numerical Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, (col, ax) in enumerate(zip(num_cols, axes.flatten())):
        sns.histplot(df[col], bins=10, kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    fig.tight_layout()
    plots['numerical_dist'] = save_plot_to_base64()

    # Skewness and Kurtosis
    skewness = stats.skew(df[num_cols])
    kurtosis = stats.kurtosis(df[num_cols]) + 3
    print(f"Skewness: {skewness}, Kurtosis: {kurtosis}")

    # Correlation Analysis
    corr_pairs = [('Year', 'Financial Loss (in Million $)'), ('Number of Affected Users', 'Financial Loss (in Million $)'),
                  ('Financial Loss (in Million $)', 'Incident Resolution Time (in Hours)'), ('Number of Affected Users', 'Year'),
                  ('Number of Affected Users', 'Incident Resolution Time (in Hours)')]
    correlations = {}
    for x, y in corr_pairs:
        corr_val = df[x].corr(df[y])
        correlations[f"{x} vs {y}"] = f"{corr_val:.2f}"
        print(f"Correlation {x} vs {y}: {corr_val:.2f}")

    # Pair Plot
    sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.5})
    plt.suptitle('Relationships Between Numerical Variables', y=1.02)
    plots['pairplot'] = save_plot_to_base64()

    # Non-linear correlations
    spearman_corr, _ = spearmanr(df['Year'], df['Financial Loss (in Million $)'])
    kendall_corr, _ = kendalltau(df['Year'], df['Financial Loss (in Million $)'])
    print(f"Spearman Correlation: {spearman_corr:.2f}")
    print(f"Kendall Correlation: {kendall_corr:.2f}")

    # Polynomial Regression
    X, y = df[['Year']], df['Financial Loss (in Million $)']
    X_poly = PolynomialFeatures(degree=2).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Polynomial Regression MSE: {mse:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5)
    plt.plot(X, model.predict(X_poly), color='red', linewidth=2)
    plt.title('Polynomial Regression')
    plt.xlabel('Year')
    plt.ylabel('Financial Loss (in Million $)')
    plots['poly_regression'] = save_plot_to_base64()

    # Chi-Square Tests
    chi_tests = [
        ('Attack Type', 'Country'), ('Attack Type', 'Attack Source'),
        ('Defense Mechanism Used', 'Security Vulnerability Type'), ('Target Industry', 'Security Vulnerability Type')
    ]
    chi_results = {}
    for idx, (x, y) in enumerate(chi_tests):
        table = pd.crosstab(df[x], df[y])
        chi2, p, _, _ = chi2_contingency(table)
        chi_results[f"{x} vs {y}"] = f"chi2={chi2:.2f}, p={p:.4f}"
        print(f"Chi-Square {x} vs {y}: chi2={chi2:.2f}, p-value={p:.2f}")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(table, annot=True, cmap='coolwarm', fmt='d')
        plt.title(f'Heatmap of {x} vs {y}')
        plots[f'chi_heatmap_{idx}'] = save_plot_to_base64()

    # ANOVA
    losses = [df.loc[df['Attack Type'] == at, 'Financial Loss (in Million $)'] for at in df['Attack Type'].unique()]
    f_stat, p = f_oneway(*losses)
    print(f"ANOVA Attack Type vs Financial Loss: F={f_stat:.2f}, p={p:.4f}")

    # Trend Visuals
    plt.figure(figsize=(12, 6))
    trend_data = df.groupby(['Year', 'Attack Type'], observed=False).size().reset_index(name='Count')
    sns.lineplot(data=trend_data, x='Year', y='Count', hue='Attack Type', marker='o', palette='Set2')
    plt.title('Attack Types Over Years')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plots['attack_trend'] = save_plot_to_base64()

    # Vulnerabilities by Industry Heatmap
    plt.figure(figsize=(12, 6))
    pivot_data = df.pivot_table(index='Target Industry', columns='Security Vulnerability Type', aggfunc='size', fill_value=0)
    sns.heatmap(pivot_data, annot=True, fmt="d", cmap="YlGnBu")
    plt.title('Vulnerabilities by Industry')
    plots['vuln_industry'] = save_plot_to_base64()

    # Average Financial Loss by Attack Type
    plt.figure(figsize=(12, 6))
    avg_loss = df.groupby('Attack Type', observed=False)['Financial Loss (in Million $)'].mean().reset_index()
    ax = sns.barplot(data=avg_loss, x='Attack Type', y='Financial Loss (in Million $)', palette='coolwarm')
    for bar in ax.patches:
        if bar.get_height():
            ax.annotate(f'{int(bar.get_height())}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                       ha='center', va='bottom')
    plt.title('Average Financial Loss by Attack Type')
    plt.xticks(rotation=45)
    plots['avg_loss'] = save_plot_to_base64()

    # Resolution Time by Attack Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Attack Type', y='Incident Resolution Time (in Hours)', palette='Set2')
    plt.title('Resolution Time by Attack Type')
    plt.xticks(rotation=45)
    plots['resolution_time'] = save_plot_to_base64()

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

    model_results = {}
    print("\n--- Model Evaluation ---")
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_results[name] = {
            'accuracy': f"{accuracy_score(y_test, y_pred):.2f}",
            'precision': f"{precision_score(y_test, y_pred, average='macro'):.2f}",
            'recall': f"{recall_score(y_test, y_pred, average='macro'):.2f}",
            'f1': f"{f1_score(y_test, y_pred, average='macro'):.2f}"
        }
        print(f"{name}: Accuracy={model_results[name]['accuracy']}, Precision={model_results[name]['precision']}, "
              f"Recall={model_results[name]['recall']}, F1={model_results[name]['f1']}")

    context = {
        'plots': plots,
        'correlations': correlations,
        'chi_results': chi_results,
        'model_results': model_results,
        'spearman': f"{spearman_corr:.2f}",
        'kendall': f"{kendall_corr:.2f}",
        'mse': f"{mse:.2f}",
        'anova': f"F={f_stat:.2f}, p={p:.4f}"
    }

    return render(request, "users/graph.html", context)