import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns

def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def transform_data(X, transformation="standardize"):
    if transformation == "standardize":
        scaler = StandardScaler()
    elif transformation == "normalize":
        scaler = MinMaxScaler()
    elif transformation == "log":
        return np.log1p(X)  # Log transform
    elif transformation == "polynomial":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        return poly.fit_transform(X)
    else:
        raise ValueError("Unsupported transformation type!")
    return scaler.fit_transform(X)

def fit_elasticnet(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    return model

def feature_importance_rfe(X_train, y_train, n_features_to_select=10):
    rfe = RFE(ElasticNet(), n_features_to_select=n_features_to_select)
    rfe.fit(X_train, y_train)
    return rfe.support_, rfe.ranking_

def plot_coefficients_pvalues(features, coefficients, pvalues, rankings):
    results_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'P-value': pvalues,
        'RFE Ranking': rankings
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # Coefficients plot
    sns.barplot(x='Feature', y='Coefficient', data=results_df, ax=ax1)
    ax1.set_title("Feature Coefficients")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # P-values plot
    sns.barplot(x='Feature', y='P-value', data=results_df, ax=ax2)
    ax2.set_title("Feature P-values")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    # RFE rankings plot
    sns.barplot(x='Feature', y='RFE Ranking', data=results_df, ax=ax3)
    ax3.set_title("RFE Rankings")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    sns.despine()
    plt.show()

def plot_transformations(data, transformations):
    for transformation in transformations:
        transformed_data = transform_data(data.copy(), transformation=transformation)
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=transformed_data.flatten(), fill=True)
        plt.title(f'Distribution after {transformation} Transformation')
        sns.despine()
        plt.show()
