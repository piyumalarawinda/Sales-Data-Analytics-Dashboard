"""
Data loading and ingestion module.
Handles CSV, Excel, SQL, and API data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data from various sources (CSV, Excel, SQL, APIs)."""
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    @staticmethod
    def load_excel(filepath: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """Load data from Excel file."""
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise
    
    @staticmethod
    def load_sql(connection_string: str, query: str) -> pd.DataFrame:
        """Load data from SQL database."""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Error loading SQL data: {e}")
            raise
    
    @staticmethod
    def load_json(filepath: str, orient: str = 'records', **kwargs) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            df = pd.read_json(filepath, orient=orient, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
