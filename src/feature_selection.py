import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif

# Displays and sorts the correlation between the target variable and all other features.
def correlation_analysis(df, target):
    corr_matrix = df.corr()

    target_corr = corr_matrix[[target]].drop(target).sort_values(by=target, ascending=False)

    plt.figure(figsize=(8, len(target_corr)/2))  # Dynamic height based on number of features
    sns.heatmap(target_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(f'Correlation with {target} (Sorted)')
    plt.tight_layout()
    plt.show()

    print(f"\nCorrelations with target variable '{target}' (sorted):")
    print(target_corr)


# Function to plot feature importance
def feature_importance(x, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)
    importances = pd.DataFrame({'feature': x.columns, 'importance': rf.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Feature Importances')
    plt.show()

    return importances

# Function to perform recursive feature elimination
def recursive_feature_elimination(x, y, n_features_to_select=10):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(x, y)

    selected_features = x.columns[rfe.support_].tolist()
    print(f"Selected features using RFE: {selected_features}")
    return selected_features

# Function to address class imbalance
def address_class_imbalance(x, y):
    smote = SMOTE(random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)

    print("Class distribution before SMOTE:")
    print(y.value_counts(normalize=True))
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts(normalize=True))

    return x_resampled, y_resampled

# Function to calculate VIF
def calculate_vif(x):
    vif_data = pd.DataFrame()
    vif_data["feature"] = x.columns
    vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

    print(vif_data)
    return vif_data

# Function to evaluate model using cross-validation
def evaluate_model_with_cv(x, y, model, cv=5):
    scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    return scores

# Function to perform mutual information test
def mutual_information_test(x, y):
    mi_scores = mutual_info_classif(x, y)
    mi_df = pd.DataFrame({'Feature': x.columns, 'Mutual Information Score': mi_scores})
    mi_df = mi_df.sort_values('Mutual Information Score', ascending=False)

    print("Mutual Information Scores:")
    print(mi_df)

    return mi_df

# Function to plot mutual information
def plot_mutual_information(mi_df, top_n=20):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Mutual Information Score', y='Feature', data=mi_df.head(top_n))
    plt.title(f"Top {top_n} Features by Mutual Information")
    plt.tight_layout()
    plt.show()

# Function to perform PCA analysis
def pca_analysis(x, n_components=5):
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x)

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by {n_components} components: {explained_variance}")

    return x_pca, explained_variance

# Function to perform ANOVA F-test
def anova_test(x, y):
    f_values, p_values = f_classif(x, y)
    anova_df = pd.DataFrame({'Feature': x.columns, 'F-Value': f_values, 'p-Value': p_values})
    anova_df = anova_df.sort_values('F-Value', ascending=False)

    print("ANOVA F-Test Results:")
    print(anova_df)

    return anova_df
