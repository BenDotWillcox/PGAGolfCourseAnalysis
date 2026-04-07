import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

CATEGORICAL_COLUMNS = ["greens", "fairways", "rough", "soil"]
DROP_COLUMNS = ["course", "fw_diff", "rgh_diff", "non_rgh_diff"]


def load_raw_data():
    course_df = pd.read_csv(DATA_DIR / "dg_course_table.csv")
    grass_df = pd.read_csv(DATA_DIR / "grass_data.csv")
    return course_df, grass_df


def build_feature_matrix(course_df, grass_df):
    """Merge course stats with grass/course characteristics, encode, and scale."""
    # One-hot encode categorical grass/soil features
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(grass_df[CATEGORICAL_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
    )

    # Merge numeric grass features
    grass_numeric = grass_df.drop(columns=["course"] + CATEGORICAL_COLUMNS)

    # Combine everything
    combined = pd.concat([course_df, grass_numeric, encoded_df], axis=1)

    labels = combined["course"]
    drop_cols = [c for c in DROP_COLUMNS if c in combined.columns]
    numeric_data = combined.drop(columns=drop_cols)

    # Replace 'undefined' strings with NaN then fill with column median
    numeric_data = numeric_data.apply(pd.to_numeric, errors="coerce")
    numeric_data = numeric_data.fillna(numeric_data.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    return X_scaled, labels, numeric_data.columns.tolist(), scaler, encoder


def pca_variance_analysis(X_scaled):
    """Run full PCA to report cumulative variance explained (for diagnostics)."""
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    return pca_full, cumulative
