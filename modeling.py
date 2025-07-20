import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _build_estimators():
    """Return dictionary of models with their param grids."""
    return {
        "Random Forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
            },
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
            },
        ),
    }


def train_model(
    df: pd.DataFrame,
    feature_columns,
    target_column: str,
):
    """Automatically selects and trains the best model using GridSearchCV."""
    X = df[feature_columns]
    y = df[target_column]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),,
            ("num", "passthrough", num_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model = None
    best_score = -1
    best_name = ""
    best_params = None

    for name, (estimator, param_grid) in _build_estimators().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", estimator)])

        try:
            grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, error_score='raise')
            grid.fit(X_train, y_train)
            score = grid.best_score_

            if score > best_score:
                best_model = grid.best_estimator_
                best_score = score
                best_name = name
                best_params = grid.best_params_

        except Exception as e:
            print(f"⚠️ Skipping {name} due to error: {str(e)}")

    if best_model is None:
        raise RuntimeError("❌ All model trainings failed. Please check your data.")

    return best_model, X_test, y_test, best_params, best_score, best_name


def get_feature_importance(model: Pipeline):
    """Return feature importance dataframe for the trained pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)

    # Determine importance
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coefs = np.abs(clf.coef_)
        if coefs.ndim > 1:
            importance = np.mean(coefs, axis=0)
        else:
            importance = coefs
    else:
        importance = np.zeros(len(feature_names))

    if len(feature_names) != len(importance):
        raise ValueError("Mismatch between feature names and importance scores")

    return pd.DataFrame({"Feature": feature_names, "Importance": importance})
