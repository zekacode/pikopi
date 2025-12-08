from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_data(df, features):
    """
    Fungsi untuk menyiapkan X, y, dan ColumnTransformer
    """
    X = df[features + [c for c in ['Processing Method','Variety'] if c in df.columns]]
    y = df['Total Cup Points'] if 'Total Cup Points' in df.columns else np.zeros(len(df))

    numeric_features = features
    categorical_features = [c for c in ['Processing Method','Variety'] if c in df.columns]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor

def benchmark_models(X_train, X_test, y_train, y_test, models_dict):
    """
    Fungsi untuk benchmark model (opsional)
    """
    results = {}
    for name, model in models_dict.items():
        # Ubah dense matrix jika sparse
        X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
        X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }
    results_df = pd.DataFrame(results).T.sort_values(by='RMSE')
    best_model_name = results_df.index[0]
    best_model = models_dict[best_model_name]
    return best_model, results_df
