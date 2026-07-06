"""
Data processing and transformation module.
Handles cleaning, feature engineering, and data validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data cleaning, transformation, and feature engineering."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """Identify columns with high missing values."""
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > threshold]
        return high_missing.to_dict()
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'drop', 
                             fill_value: Union[int, float, str] = None) -> pd.DataFrame:
        """Handle missing values via drop or fill strategies."""
        df_clean = df.copy()
        
        if method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'fill_mean':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif method == 'fill_median':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif method == 'fill_value':
            df_clean = df_clean.fillna(fill_value)
        
        logger.info(f"Handled missing values using method: {method}")
        return df_clean
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        df_clean = df.drop_duplicates(subset=subset, keep='first')
        logger.info(f"Removed {len(df) - len(df_clean)} duplicates")
        return df_clean
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method."""
        df_clean = df.copy()
        
        if method == 'iqr':
            Q1 = df_clean[columns].quantile(0.25)
            Q3 = df_clean[columns].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((df_clean[columns] < (Q1 - threshold * IQR)) | 
                     (df_clean[columns] > (Q3 + threshold * IQR))).any(axis=1)
            df_clean = df_clean[mask]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_clean[columns]))
            mask = (z_scores < threshold).all(axis=1)
            df_clean = df_clean[mask]
        
        logger.info(f"Removed {len(df) - len(df_clean)} outliers using {method}")
        return df_clean
    
    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str], 
                          method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical variables."""
        df_encoded = df.copy()
        encoders = {}
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
        
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        
        logger.info(f"Encoded {len(columns)} categorical columns using {method} encoding")
        return df_encoded, encoders
    
    @staticmethod
    def normalize_features(df: pd.DataFrame, columns: List[str], 
                          method: str = 'standard') -> Tuple[pd.DataFrame, object]:
        """Normalize numeric features."""
        df_normalized = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
            df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        
        logger.info(f"Normalized {len(columns)} columns using {method} scaling")
        return df_normalized, scaler
    
    @staticmethod
    def create_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Extract datetime features from a datetime column."""
        df_feat = df.copy()
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])
        
        df_feat[f'{datetime_col}_year'] = df_feat[datetime_col].dt.year
        df_feat[f'{datetime_col}_month'] = df_feat[datetime_col].dt.month
        df_feat[f'{datetime_col}_day'] = df_feat[datetime_col].dt.day
        df_feat[f'{datetime_col}_dayofweek'] = df_feat[datetime_col].dt.dayofweek
        df_feat[f'{datetime_col}_hour'] = df_feat[datetime_col].dt.hour
        df_feat[f'{datetime_col}_quarter'] = df_feat[datetime_col].dt.quarter
        
        logger.info(f"Created {6} datetime features from {datetime_col}")
        return df_feat
