import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def normalize_series(series):
    """Normalize a time series using MinMaxScaler"""
    values = series.values
    values = values.reshape((len(values), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    normalized = pd.Series(normalized.flatten(), index=series.index)
    return normalized


def remap_indexes_after_nan_removal(series):
    """Remap the indexes of a series after nan removal
    This is useful for remapping the indexes of a series after nan removal.
    """
    data = series.to_numpy()
    df = pd.DataFrame({"value": data, "original_index": range(len(data))})
    df_dropna = df.dropna().reset_index(drop=True)
    mapping_df = pd.DataFrame(
        {
            "new_index": df_dropna.index,
            "original_index": df_dropna["original_index"],
        }
    )
    index_mapping = np.array(list(mapping_df.itertuples(index=False, name=None)))[:, 1]
    return index_mapping
