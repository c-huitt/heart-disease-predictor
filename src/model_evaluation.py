import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


# Function to interpret the model
def interpret_model(pipeline, X, y, feature_names, model_name):
    # Extract the actual model from the pipeline and get transformed feature names
    if isinstance(pipeline, Pipeline):
        model = pipeline.named_steps['model']
        X_transformed = pipeline[:-1].transform(X)
        if hasattr(pipeline[:-1], 'get_feature_names_out'):
            feature_names = pipeline[:-1].get_feature_names_out()
        elif hasattr(X_transformed, 'columns'):
            feature_names = X_transformed.columns.tolist()
    else:
        model = pipeline
        X_transformed = X

    # Feature Importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError(f"Cannot determine feature importances for model type: {type(model)}")

    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # Visualize Feature Importances
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index)
    plt.title(f'Feature Importances in {model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    # Permutation Importance
    perm_importance = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42)
    perm_importances = pd.Series(perm_importance.importances_mean, index=feature_names).sort_values(ascending=False)

    # Visualize Permutation Importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=perm_importances.values, y=perm_importances.index)
    plt.title(f'Permutation Importance in {model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    return feature_importances, perm_importances
