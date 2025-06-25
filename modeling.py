import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _build_estimator(name: str):
    if name == "Random Forest":
        estimator = RandomForestClassifier(random_state=42, class_weight="balanced")
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
        }
    elif name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1000)
        param_grid = {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        }
    else:
        raise ValueError(f"Unsupported model type: {name}")
    return estimator, param_grid


def train_model(
    df: pd.DataFrame,
    feature_columns,
    target_column: str,
    model_name: str = "Random Forest",
    grid_search: bool = False,
):
    """Train a model using a preprocessing pipeline."""
    X = df[feature_columns]
    y = df[target_column]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    estimator, param_grid = _build_estimator(model_name)

    pipe = Pipeline([("preprocessor", preprocessor), ("clf", estimator)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_params = None
    cv_score = None

    if grid_search:
        try:
            grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, error_score='raise')
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            best_params = grid.best_params_
            cv_score = grid.best_score_
        except Exception as e:
            raise RuntimeError(f"❌ GridSearchCV failed: {str(e)}")
    else:
        scores = cross_val_score(pipe, X_train, y_train, cv=5)
        cv_score = scores.mean()
        model = pipe.fit(X_train, y_train)

    return model, X_test, y_test, best_params, cv_score


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

    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coef = clf.coef_[0]
        importance = abs(coef)
    else:
        importance = [0] * len(feature_names)

    return pd.DataFrame({"Feature": feature_names, "Importance": importance})
