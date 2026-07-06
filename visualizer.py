"""
Machine learning models and training module.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class MLModels:
    """Train and evaluate machine learning models."""
    
    @staticmethod
    def train_test_split_data(X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, 
                              random_state: int = 42) -> Tuple:
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                                  **kwargs) -> LogisticRegression:
        """Train logistic regression model."""
        model = LogisticRegression(**kwargs)
        model.fit(X_train, y_train)
        logger.info("Logistic Regression model trained")
        return model
    
    @staticmethod
    def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                           n_estimators: int = 100, **kwargs) -> RandomForestClassifier:
        """Train random forest classifier."""
        model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        model.fit(X_train, y_train)
        logger.info(f"Random Forest model trained with {n_estimators} trees")
        return model
    
    @staticmethod
    def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                     n_estimators: int = 100, **kwargs) -> xgb.XGBClassifier:
        """Train XGBoost classifier."""
        model = xgb.XGBClassifier(n_estimators=n_estimators, **kwargs)
        model.fit(X_train, y_train)
        logger.info(f"XGBoost model trained with {n_estimators} estimators")
        return model
    
    @staticmethod
    def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series,
                               **kwargs) -> LinearRegression:
        """Train linear regression model."""
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        logger.info("Linear Regression model trained")
        return model
    
    @staticmethod
    def evaluate_classification(model, X_test: pd.DataFrame, 
                               y_test: pd.Series) -> Dict:
        """Evaluate classification model performance."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"Classification metrics: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1']:.4f}")
        return metrics
    
    @staticmethod
    def evaluate_regression(model, X_test: pd.DataFrame,
                           y_test: pd.Series) -> Dict:
        """Evaluate regression model performance."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mae': np.mean(np.abs(y_test - y_pred))
        }
        
        logger.info(f"Regression metrics: R²={metrics['r2']:.4f}, "
                   f"RMSE={metrics['rmse']:.4f}")
        return metrics
    
    @staticmethod
    def cross_validate(model, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5) -> np.ndarray:
        """Perform cross-validation."""
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        logger.info(f"Cross-validation scores (cv={cv}): mean={scores.mean():.4f}, "
                   f"std={scores.std():.4f}")
        return scores
    
    @staticmethod
    def feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Top 5 features: {importance_df.head(5)['feature'].tolist()}")
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return None
