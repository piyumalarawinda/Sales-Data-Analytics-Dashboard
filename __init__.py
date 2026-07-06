"""
Unit tests for data processing and ML models.
Run: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from src.data_processor import DataProcessor
from src.analytics import Analytics
from src.ml_models import MLModels


class TestDataProcessor:
    """Test data processing functions."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [100, 100, 200, 200, 300],
        })
    
    def test_handle_missing_values(self, sample_df):
        df_clean = DataProcessor.handle_missing_values(sample_df, method='fill_mean')
        assert not df_clean.isnull().any().any()
    
    def test_remove_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df])
        df_unique = DataProcessor.remove_duplicates(df_dup)
        assert len(df_unique) == len(sample_df)


class TestAnalytics:
    """Test statistical analysis functions."""
    
    @pytest.fixture
    def sample_data(self):
        return np.random.normal(0, 1, 100)
    
    def test_descriptive_stats(self, sample_data):
        df = pd.DataFrame({'value': sample_data})
        stats = Analytics.descriptive_stats(df)
        assert 'value' in stats.index


class TestMLModels:
    """Test ML model training and evaluation."""
    
    @pytest.fixture
    def sample_dataset(self):
        X = pd.DataFrame(np.random.rand(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    def test_train_test_split(self, sample_dataset):
        X, y = sample_dataset
        X_train, X_test, y_train, y_test = MLModels.train_test_split_data(X, y)
        assert len(X_train) + len(X_test) == len(X)
    
    def test_logistic_regression(self, sample_dataset):
        X, y = sample_dataset
        X_train, X_test, y_train, y_test = MLModels.train_test_split_data(X, y)
        model = MLModels.train_logistic_regression(X_train, y_train, max_iter=1000)
        metrics = MLModels.evaluate_classification(model, X_test, y_test)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
