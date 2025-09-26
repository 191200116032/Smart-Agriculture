import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crop_trainer")

FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def find_dataset(candidate: str = "dataset/crop_recommendation/crop_recommendation.csv") -> Path:
    """Locate dataset automatically from likely locations."""
    path_candidates = [
        Path(candidate),
        Path(__file__).resolve().parent.joinpath(candidate),
        Path(__file__).resolve().parents[1].joinpath(candidate),
        Path(__file__).resolve().parents[2].joinpath(candidate),
    ]
    for p in path_candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Dataset not found. Tried these locations:\n" +
        "\n".join([str(p.resolve()) for p in path_candidates])
    )


def validate_input_ranges(data: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme values to allowed ranges."""
    logger.info("Dataset statistics:\n%s", data.describe().transpose())
    ranges = {
        'N': (0, 200),
        'P': (0, 150),
        'K': (0, 300),
        'temperature': (5, 45),
        'humidity': (10, 100),
        'ph': (3, 10),
        'rainfall': (20, 3000)
    }
    for feature, (min_val, max_val) in ranges.items():
        if feature in data.columns:
            original = data[feature].copy()
            data[feature] = data[feature].clip(min_val, max_val)
            n_clipped = (original != data[feature]).sum()
            if n_clipped > 0:
                logger.warning("%s: %d outliers clipped", feature, n_clipped)
    return data


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to capture environmental interactions."""
    X = X.copy()
    if 'temperature' in X.columns and 'humidity' in X.columns:
        X['temp_humidity'] = X['temperature'] * X['humidity']
    if 'rainfall' in X.columns and 'N' in X.columns:
        X['rainfall_per_N'] = X['rainfall'] / (X['N'] + 1e-5)
    return X


def train_and_save_model(
        dataset_path: Path,
        model_dir: Path,
        model_filename: str = "crop_recommendation_saved_model.pkl",
        n_iter: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        version: bool = False
):
    dataset_path = Path(dataset_path).resolve()
    model_dir = Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    data = pd.read_csv(dataset_path)
    logger.info("Dataset shape: %s", data.shape)

    data = validate_input_ranges(data)

    if data.isnull().sum().any():
        logger.warning("Missing values found — dropping rows with NA")
        data = data.dropna()

    if 'label' not in data.columns:
        raise KeyError(f"'label' column not found in dataset. Found columns: {list(data.columns)}")

    X = data.drop('label', axis=1)
    y = data['label']

    # Add engineered interaction features
    X = add_interaction_features(X)
    FEATURE_NAMES_EXTENDED = list(X.columns)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(random_state=random_state,
                                              class_weight='balanced',
                                              oob_score=True))
    ])

    param_dist = {
        'classifier__n_estimators': randint(200, 500),
        'classifier__max_depth': randint(10, 50),
        'classifier__min_samples_split': randint(2, 15),
        'classifier__min_samples_leaf': randint(1, 8),
        'classifier__max_features': ['sqrt', 'log2', 0.8],
        'classifier__bootstrap': [True, False]
    }

    logger.info("Starting hyperparameter search (n_iter=%d, n_jobs=%s)...", n_iter, n_jobs)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='accuracy',
        cv=5,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    logger.info("Best parameters: %s", search.best_params_)
    logger.info("Best CV score (search): %.4f", search.best_score_)

    cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
    logger.info("10-fold CV accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Test accuracy: %.4f", acc)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

    rf_model = best_model.named_steps['classifier']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info("Feature importance:\n%s", feature_importance)

    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=random_state)
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    logger.info("Permutation importance:\n%s", perm_df)

    if hasattr(rf_model, 'oob_score_'):
        logger.info("Out-of-bag score: %.4f", rf_model.oob_score_)

    if acc < 0.90:
        logger.warning("Accuracy below 90%% — consider more tuning or different models.")

    # Prepare metadata
    model_metadata = {
        'model': best_model,
        'feature_names': FEATURE_NAMES_EXTENDED,
        'accuracy': acc,
        'cv_score': cv_scores.mean(),
        'best_params': search.best_params_,
        'feature_importance': feature_importance.to_dict('records'),
        'classes': list(best_model.classes_)
    }

    if version:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = model_dir / f"{Path(model_filename).stem}_{ts}{Path(model_filename).suffix}"
    else:
        out_file = model_dir / model_filename

    joblib.dump(model_metadata, out_file)
    logger.info("Model saved to: %s", out_file)

    return best_model, feature_importance
