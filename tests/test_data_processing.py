import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_data,
    handle_missing_values,
    optimize_memory,
    encode_categoricals,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [10.0, None, 15.0, 20.0],
        'weight': [30.0, 40.0, None, 50.0],
        'gender': ['M', 'F', 'M', None],
        'survival_status': [1, 0, 1, 0]
    })


class TestMissingValues:
    def test_no_missing_after_handle(self, sample_df):
        result = handle_missing_values(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_shape_preserved(self, sample_df):
       result = handle_missing_values(sample_df)        
