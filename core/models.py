import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

MODEL_PATH = Path("core/saved_model.pkl")
FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def validate_input_ranges(data):
    """Validate and log data ranges to identify potential issues"""
    print("\nDataset Statistics:")
    print(data.describe())

    # Define reasonable ranges for crop growing conditions
    ranges = {
        'N': (0, 200),
        'P': (0, 150),
        'K': (0, 300),
        'temperature': (5, 45),
        'humidity': (10, 100),
        'ph': (3, 10),
        'rainfall': (20, 3000)
    }

    print("\nOutlier Analysis:")
    for feature, (min_val, max_val) in ranges.items():
        if feature in data.columns:
            outliers = data[(data[feature] < min_val) | (data[feature] > max_val)]
            if len(outliers) > 0:
                print(f"{feature}: {len(outliers)} outliers ({len(outliers) / len(data) * 100:.1f}%)")


def train_and_save_model(dataset_path: str = "dataset/crop_recommendation.csv"):
    try:
        # Load dataset
        print("Loading dataset...")
        data = pd.read_csv(dataset_path)
        print(f"Dataset shape: {data.shape}")

        # Validate data
        validate_input_ranges(data)

        # Check for missing values
        if data.isnull().sum().any():
            print("\nWarning: Missing values found!")
            print(data.isnull().sum())
            data = data.dropna()

        # Check class distribution
        print(f"\nClass distribution:")
        print(data['label'].value_counts().head(10))

        X = data.drop('label', axis=1)
        y = data['label']

        # Ensure we have the expected features
        expected_features = set(FEATURE_NAMES)
        actual_features = set(X.columns)
        if expected_features != actual_features:
            print(f"Warning: Feature mismatch!")
            print(f"Expected: {expected_features}")
            print(f"Actual: {actual_features}")

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Use RobustScaler instead of StandardScaler for better outlier handling
        pipeline = Pipeline([
            ('scaler', RobustScaler()),  # More robust to outliers
            ('classifier', RandomForestClassifier(
                random_state=42,
                class_weight='balanced',  # Handle class imbalance
                oob_score=True  # Out-of-bag scoring
            ))
        ])

        # Expanded hyperparameter search
        param_dist = {
            'classifier__n_estimators': randint(200, 500),
            'classifier__max_depth': randint(10, 50),
            'classifier__min_samples_split': randint(2, 15),
            'classifier__min_samples_leaf': randint(1, 8),
            'classifier__max_features': ['sqrt', 'log2', 0.8],
            'classifier__bootstrap': [True, False]
        }

        print("\nStarting hyperparameter optimization...")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=50,  # More iterations for better results
            scoring='accuracy',
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.4f}")

        # Cross-validation on full dataset
        cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
        print(f"10-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nFinal Model Performance:")
        print(f"Test Accuracy: {acc:.4f}")

        # Detailed classification report
        print(f"\nClassification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))

        # Feature importance analysis
        print(f"\nFeature Importance Analysis:")
        rf_model = best_model.named_steps['classifier']
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(feature_importance)

        # Permutation importance (more reliable)
        print(f"\nPermutation Importance (more reliable):")
        perm_importance = permutation_importance(
            best_model, X_test, y_test, n_repeats=5, random_state=42
        )
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        print(perm_df)

        # Model diagnostics
        print(f"\nModel Diagnostics:")
        if hasattr(rf_model, 'oob_score_'):
            print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")

        # Check for potential issues
        if acc < 0.90:
            print("⚠ Warning: Accuracy below 90%. Potential issues:")
            print("  - Check data quality and feature engineering")
            print("  - Consider ensemble methods or different algorithms")
            print("  - Verify label quality and consistency")
        elif acc < 0.95:
            print("⚠ Moderate accuracy. Consider further tuning.")
        else:
            print("✓ High accuracy achieved!")

        # Save model with metadata
        model_metadata = {
            'model': best_model,
            'feature_names': list(X.columns),
            'accuracy': acc,
            'cv_score': cv_scores.mean(),
            'best_params': search.best_params_,
            'feature_importance': feature_importance.to_dict('records'),
            'classes': list(best_model.classes_)
        }

        joblib.dump(model_metadata, MODEL_PATH)
        print(f"\n✓ Model saved at {MODEL_PATH} with accuracy {acc:.4f}")

        return best_model, feature_importance

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


def load_model():
    """Load model with error handling"""
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        model_data = joblib.load(MODEL_PATH)

        # Handle both old format (just model) and new format (dict with metadata)
        if isinstance(model_data, dict):
            print(f"Loaded model with accuracy: {model_data['accuracy']:.4f}")
            return model_data['model']
        else:
            print("Loaded legacy model format")
            return model_data

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def predict_crop(model, input_data):
    """Make prediction with input validation"""
    try:
        # Validate input
        if isinstance(input_data, dict):
            # Convert dict to DataFrame
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        else:
            input_df = input_data

        # Check for extreme values
        extreme_checks = {
            'temperature': (5, 45),
            'rainfall': (20, 3000),
            'humidity': (10, 100),
            'ph': (3, 10)
        }

        warnings = []
        for feature, (min_val, max_val) in extreme_checks.items():
            if feature in input_df.columns:
                value = input_df[feature].iloc[0]
                if value < min_val or value > max_val:
                    warnings.append(f"{feature} ({value}) is outside typical range ({min_val}-{max_val})")

        if warnings:
            print("⚠ Input validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = [(model.classes_[i], probabilities[i]) for i in top_3_idx]

        return {
            'prediction': prediction,
            'confidence': max(probabilities),
            'top_3': top_3_crops,
            'warnings': warnings
        }

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise