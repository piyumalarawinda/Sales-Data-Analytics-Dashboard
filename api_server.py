"""
Statistical analysis and exploratory data analysis module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class Analytics:
    """Perform statistical analysis and generate insights."""
    
    @staticmethod
    def descriptive_stats(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Calculate descriptive statistics."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        return df[columns].describe().T
    
    @staticmethod
    def correlation_analysis(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Calculate correlation matrix."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        return df[columns].corr()
    
    @staticmethod
    def distribution_test(data: np.ndarray, test: str = 'shapiro') -> Tuple[float, float]:
        """Test data normality (Shapiro-Wilk or Anderson-Darling)."""
        if test == 'shapiro':
            stat, p_value = stats.shapiro(data)
        elif test == 'anderson':
            result = stats.anderson(data)
            stat, p_value = result.statistic, result.significance_level[0] / 100
        else:
            raise ValueError("Test must be 'shapiro' or 'anderson'")
        
        logger.info(f"Distribution test ({test}): statistic={stat:.4f}, p-value={p_value:.4f}")
        return stat, p_value
    
    @staticmethod
    def hypothesis_test(data1: np.ndarray, data2: np.ndarray, 
                       test: str = 'ttest') -> Tuple[float, float]:
        """Perform hypothesis testing (t-test, Mann-Whitney U)."""
        if test == 'ttest':
            stat, p_value = stats.ttest_ind(data1, data2)
        elif test == 'mannwhitneyu':
            stat, p_value = stats.mannwhitneyu(data1, data2)
        else:
            raise ValueError("Test must be 'ttest' or 'mannwhitneyu'")
        
        logger.info(f"Hypothesis test ({test}): statistic={stat:.4f}, p-value={p_value:.4f}")
        return stat, p_value
    
    @staticmethod
    def group_analysis(df: pd.DataFrame, group_col: str, 
                      agg_cols: List[str] = None) -> pd.DataFrame:
        """Group-by analysis with aggregations."""
        if agg_cols is None:
            agg_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        grouped = df.groupby(group_col)[agg_cols].agg(['mean', 'median', 'std', 'count'])
        logger.info(f"Group analysis on {group_col}: {len(grouped)} groups")
        return grouped
    
    @staticmethod
    def outlier_stats(df: pd.DataFrame, columns: List[str] = None) -> Dict:
        """Identify and report outliers by column."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_info[col] = len(outliers)
        
        logger.info(f"Identified outliers across {len(columns)} columns")
        return outlier_info
