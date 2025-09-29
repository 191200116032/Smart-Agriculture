import joblib
import logging
import pandas as pd
from pathlib import Path

from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("yield_trainer")

MODEL_PATH = Path("models/crop_yield_saved_model.pkl")


def train_and_save_yield_model(dataset_path: str = "dataset/crop_yield/yield_df.csv", n_iter: int = 30):
    logger = logging.getLogger("yield_trainer")
    logging.basicConfig(level=logging.INFO)

    MODEL_PATH = Path("models/crop_yield_saved_model.pkl")

    logger.info("Loading dataset...")
    df = pd.read_csv(dataset_path)

    # Rename columns to match expected names
    df = df.rename(columns={
        'Area': 'State',
        'Item': 'Crop',
        'hg/ha_yield': 'Yield',
        'average_rain_fall_mm_per_year': 'Rainfall',
        'pesticides_tonnes': 'Pesticides',
        'avg_temp': 'Temperature'
    })

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['State', 'Crop'], drop_first=True)

    X = df.drop('Yield', axis=1)
    y = df['Yield']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    param_dist = {
        'regressor__n_estimators': randint(150, 400),
        'regressor__max_depth': randint(5, 30),
        'regressor__min_samples_split': randint(2, 10),
        'regressor__min_samples_leaf': randint(1, 5),
        'regressor__max_features': ['sqrt', 'log2']
    }

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter,
                                cv=5, scoring='r2', n_jobs=-1, verbose=1, random_state=42)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info("R²: %.4f, MSE: %.2f", r2, mse)

    joblib.dump((best_model, X.columns.tolist()), MODEL_PATH)
    logger.info("Crop yield model saved at %s", MODEL_PATH)
