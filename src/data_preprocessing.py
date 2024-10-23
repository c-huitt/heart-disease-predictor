from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import os
import pandas as pd

# Loads the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Saves the data
def save_data(df, output_path):
    df.to_csv(output_path, index=False)

# Handles missing values for both categorical columns
def impute_categorical_missing_data(df, passed_col, missing_data_cols, bool_cols):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y.astype(str))

    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    print(f"The feature '{passed_col}' has been imputed with {round(acc_score * 100, 2)}% accuracy\n")

    if not df_null.empty:
        X_null = df_null.drop(passed_col, axis=1)
        for col in X_null.columns:
            if X_null[col].dtype == 'object' or X_null[col].dtype == 'category':
                X_null[col] = label_encoder.fit_transform(X_null[col].astype(str))

        for col in other_missing_cols:
            if X_null[col].isnull().sum() > 0:
                col_with_missing_values = X_null[col].values.reshape(-1, 1)
                imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
                X_null[col] = imputed_values[:, 0]

        df_null[passed_col] = rf_classifier.predict(X_null)
        if passed_col in bool_cols:
            df_null[passed_col] = df_null[passed_col].map({0: False, 1: True})

    df_combined = pd.concat([df_not_null, df_null])
    return df_combined[passed_col]

# Handles missing values for numeric columns
def impute_numeric(df, passed_col):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    missing_numeric_columns = numeric_columns[df[numeric_columns].isnull().any()]

    if passed_col in missing_numeric_columns:
        try:
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=100, random_state=42),
                max_iter=10,
                random_state=42
            )
            imputed_values = imputer.fit_transform(df[[passed_col]])

            df_imputed = df.copy()
            df_imputed[passed_col] = imputed_values.ravel()

            print(f"Numeric imputation completed for column: {passed_col}")

            X = df.drop(columns=[passed_col])
            y = df[passed_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            rf_regressor = RandomForestRegressor(random_state=42)
            rf_regressor.fit(X_train, y_train)
            y_pred = rf_regressor.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            print(f"MAE = {mae}")
            print(f"RMSE = {rmse}")
            print(f"R2 = {r2}\n")

            return df_imputed[passed_col]
        except Exception as e:
            print(f"Error during iterative imputation: {str(e)}")
            print("Falling back to mean imputation for numeric column.")
            return df[passed_col].fillna(df[passed_col].mean())
    else:
        print(f"No missing values found in numeric column: {passed_col}")
        return df[passed_col]

# Evaluate missing data imputation, displaying accuracy for categorical columns.
def evaluate_imputation(df):
    missing_data_cols = df.columns[df.isnull().any()]
    accuracy_dict = {}

    for col in missing_data_cols:
        missing_percentage = round((df[col].isnull().sum() / len(df)) * 100, 2)
        print(f"Missing Values in '{col}': {missing_percentage}%")

        if df[col].dtype == 'object':
            print(f"Imputing categorical column: {col}")
            df, acc = impute_categorical(df)
            accuracy_dict.update(acc)
        else:
            print(f"Imputing numeric column: {col}")
            df = impute_numeric(df)

    print("\nImputation completed!\n")

    print("Accuracy for imputed categorical columns:")
    for col, accuracy in accuracy_dict.items():
        print(f"Accuracy for '{col}': {round(accuracy * 100, 2)}%")

    return df

# Preprocesses the features
def feature_preprocess():

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    preprocess = ColumnTransformer(
        transformers=[
            ('num_standard', standard_scaler, numeric_cols),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocess

# Preprocesses the target variable
def target_preprocess(df):
    df['target'] = df['num']

    return df.drop('num', axis=1)

# Pipeline for preprocessing the data
def preprocess_pipeline(file_path):
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = target_preprocess(df)

    x = df.drop('target', axis=1)
    y = df['target']

    preprocessor = feature_preprocess()
    x_processed = preprocessor.fit_transform(x)

    feature_names = (numeric_cols +
                     preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features).tolist())

    x_processed_df = pd.DataFrame(x_processed, columns=feature_names)

    x_processed_df['target'] = y.values

    output_dir = os.path.dirname(file_path)
    output_filename = os.path.join(output_dir, 'preprocessed_heart_disease_data.csv')
    x_processed_df.to_csv(output_filename, index=False)

    print(f"Preprocessed data saved to: {output_filename}")

    return x_processed_df.drop('target', axis=1), y, preprocessor