"""
Example: End-to-End Data Analysis Pipeline
This script demonstrates the complete workflow.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.analytics import Analytics
from src.ml_models import MLModels
from src.visualizer import Visualizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run complete analysis pipeline."""
    
    # 1. Load Data
    logger.info("=" * 50)
    logger.info("Step 1: Loading Data")
    logger.info("=" * 50)
    
    # Example: Create synthetic dataset
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 80, 1000),
        'Income': np.random.normal(50000, 20000, 1000),
        'Education': np.random.choice([1, 2, 3, 4], 1000),
        'Score': np.random.uniform(0, 100, 1000)
    }
    df = pd.DataFrame(data)
    df['Target'] = (df['Score'] > 50).astype(int)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"\n{df.head()}")
    
    # 2. Data Processing
    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Data Processing")
    logger.info("=" * 50)
    
    # Check missing values
    missing = DataProcessor.check_missing_values(df)
    logger.info(f"Missing values: {missing}")
    
    # Remove duplicates
    df_clean = DataProcessor.remove_duplicates(df)
    logger.info(f"After duplicate removal: {df_clean.shape}")
    
    # 3. Exploratory Data Analysis
    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Exploratory Data Analysis")
    logger.info("=" * 50)
    
    # Descriptive statistics
    stats = Analytics.descriptive_stats(df_clean)
    logger.info(f"\nDescriptive Statistics:\n{stats}")
    
    # Correlation analysis
    corr = Analytics.correlation_analysis(df_clean)
    logger.info(f"\nCorrelation Matrix:\n{corr}")
    
    # Group analysis
    grouped = Analytics.group_analysis(df_clean, 'Target')
    logger.info(f"\nGroup Analysis:\n{grouped}")
    
    # 4. Feature Engineering & Encoding
    logger.info("\n" + "=" * 50)
    logger.info("Step 4: Feature Engineering")
    logger.info("=" * 50)
    
    # Normalize features
    numeric_cols = ['Age', 'Income', 'Score']
    df_normalized, scaler = DataProcessor.normalize_features(
        df_clean, numeric_cols, method='standard'
    )
    logger.info(f"Normalized features: {numeric_cols}")
    
    # 5. Machine Learning
    logger.info("\n" + "=" * 50)
    logger.info("Step 5: Machine Learning")
    logger.info("=" * 50)
    
    # Prepare features and target
    X = df_normalized[['Age', 'Income', 'Education', 'Score']]
    y = df_normalized['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = MLModels.train_test_split_data(X, y)
    
    # Train models
    logger.info("\nTraining models...")
    
    # Logistic Regression
    lr_model = MLModels.train_logistic_regression(X_train, y_train, max_iter=1000)
    lr_metrics = MLModels.evaluate_classification(lr_model, X_test, y_test)
    logger.info(f"Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, F1: {lr_metrics['f1']:.4f}")
    
    # Random Forest
    rf_model = MLModels.train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    rf_metrics = MLModels.evaluate_classification(rf_model, X_test, y_test)
    logger.info(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    # XGBoost
    xgb_model = MLModels.train_xgboost(X_train, y_train, n_estimators=100, random_state=42, use_label_encoder=False)
    xgb_metrics = MLModels.evaluate_classification(xgb_model, X_test, y_test)
    logger.info(f"XGBoost - Accuracy: {xgb_metrics['accuracy']:.4f}, F1: {xgb_metrics['f1']:.4f}")
    
    # Cross-validation
    logger.info("\nPerforming cross-validation...")
    cv_scores = MLModels.cross_validate(rf_model, X, y, cv=5)
    logger.info(f"CV Scores: {cv_scores}")
    logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    feature_imp = MLModels.feature_importance(rf_model, X.columns.tolist())
    logger.info(f"\nFeature Importance:\n{feature_imp}")
    
    # 6. Visualization
    logger.info("\n" + "=" * 50)
    logger.info("Step 6: Visualization")
    logger.info("=" * 50)
    
    viz = Visualizer()
    
    # Save correlation heatmap
    logger.info("Creating correlation heatmap...")
    fig_corr = viz.plot_correlation_heatmap(df_clean)
    fig_corr.savefig('reports/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: reports/correlation_heatmap.png")
    
    # Save confusion matrix
    logger.info("Creating confusion matrix...")
    fig_cm = viz.plot_confusion_matrix(rf_metrics['confusion_matrix'], labels=['No', 'Yes'])
    fig_cm.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info("Saved: reports/confusion_matrix.png")
    
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
