from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Split the data into train and test sets
def split_data(x, y, test_size=0.3, random_state=42):
    """Split the data into train and test sets."""
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

# Evaluate the models
def evaluate_models(X_train, X_test, y_train, y_test, models, metric_weights):
    results = {}
    best_model = None
    best_weighted_score = 0.0
    best_model_name = ""

    for name, model in models:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Perform cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        mean_cv_accuracy = scores.mean()

        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

        # Calculate the weighted score based on the metric weights
        weighted_score = (
                metric_weights['Accuracy'] * accuracy +
                metric_weights['Precision'] * precision +
                metric_weights['Recall'] * recall +
                metric_weights['F1 Score'] * f1 +
                metric_weights['ROC-AUC'] * roc_auc
        )

        # Store the metrics in a dictionary
        metrics = {
            "Cross-Validation Accuracy": mean_cv_accuracy,
            "Test Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "ROC-AUC": roc_auc,
            "Weighted Score": weighted_score

        }

        results[name] = metrics

        # Check if the current model has the best accuracy
        if weighted_score > best_weighted_score:
            best_weighted_score = weighted_score
            best_model = pipeline
            best_model_name = name

    return results, best_model, best_model_name

# Tune hyperparameters for all models
def tune_all_models(X_train, y_train, X_test, y_test, cv=5, n_iter=50, metric_weights=None):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier(verbose=-1)
    }

    param_distributions = {
        'Logistic Regression': {
            'model__C': np.logspace(-4, 4, 20),
            'model__penalty': ['l2'],
            'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        'Random Forest': {
            'model__n_estimators': np.arange(100, 1000, 100),
            'model__max_depth': np.arange(3, 15),
            'model__min_samples_split': np.arange(2, 20),
            'model__min_samples_leaf': np.arange(1, 10),
            'model__max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'model__n_estimators': np.arange(100, 1000, 100),
            'model__max_depth': np.arange(3, 10),
            'model__learning_rate': np.logspace(-3, 0, 10),
            'model__subsample': np.arange(0.5, 1.0, 0.1),
            'model__max_features': ['sqrt', 'log2', None]
        },
        'KNeighbors': {
            'model__n_neighbors': np.arange(1, 20),
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'SVM': {
            'model__C': np.logspace(-3, 3, 10),
            'model__kernel': ['rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 10))
        },
        'XGBoost': {
            'model__n_estimators': np.arange(100, 1000, 100),
            'model__max_depth': np.arange(3, 10),
            'model__learning_rate': np.logspace(-3, 0, 10),
            'model__subsample': np.arange(0.5, 1.0, 0.1),
            'model__colsample_bytree': np.arange(0.5, 1.0, 0.1),
            'model__min_child_weight': np.arange(1, 10)
        },
        'LightGBM': {
            'model__n_estimators': np.arange(100, 1000, 100),
            'model__max_depth': np.arange(3, 10),
            'model__learning_rate': np.logspace(-3, 0, 10),
            'model__num_leaves': np.arange(20, 100, 10),
            'model__subsample': np.arange(0.5, 1.0, 0.1),
            'model__colsample_bytree': np.arange(0.5, 1.0, 0.1)
        }
    }

    results = {}

    for name, model in models.items():
        print(f"\nTuning {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        try:
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions[name],
                n_iter=n_iter,
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

            random_search.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = random_search.best_estimator_.predict(X_test)
            y_pred_proba = random_search.best_estimator_.predict_proba(X_test) if hasattr(random_search.best_estimator_,
                                                                                          "predict_proba") else None

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr',
                                    average='macro') if y_pred_proba is not None else 0

            # Calculate the weighted score based on the metric weights
            weighted_score = (
                    metric_weights['Accuracy'] * accuracy +
                    metric_weights['Precision'] * precision +
                    metric_weights['Recall'] * recall +
                    metric_weights['F1 Score'] * f1 +
                    metric_weights['ROC-AUC'] * roc_auc
            )

            results[name] = {
                'best_params': random_search.best_params_,
                'best_score': weighted_score,  # Store the weighted score
                'best_estimator': random_search.best_estimator_,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
            }

            print(f"Best parameters for {name}:")
            print(results[name]['best_params'])
            print(f"Weighted score: {weighted_score:.4f}")
        except Exception as e:
            print(f"An error occurred while tuning {name}: {str(e)}")
            results[name] = {
                'error': str(e)
            }

    return results

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
