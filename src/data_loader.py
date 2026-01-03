"""
Data loader module for loading datasets based on YAML configuration.
"""

import yaml
import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Dataset loader with PyTorch-style interface and automatic target selection.

    Usage:
        loader = DataLoader('configs/scanb.yaml')

        # Access full dataset
        X = loader.X            # All features
        y = loader.y            # Current target
        report = loader.report  # Loading statistics

        # PyTorch-style indexing
        X_sample, y_sample = loader[0]      # Single sample
        X_batch, y_batch = loader[:10]      # Batch of 10

        # Iterate through dataset
        for X_i, y_i in loader:
            # Process each sample
            pass

        # Switch target
        y_er = loader.set_target('ER')

        # Dataset info
        print(f"Dataset size: {len(loader)}")
    """

    def __init__(self, config_path: str):
        """
        Load dataset based on YAML configuration.

        Args:
            config_path: Path to the YAML config file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.X, self.metadata, self.report = self._load_dataset()
        self._target_column = self.config['target_column']

        print(f"Loaded dataset: {self.config['name']}")
        print(f"  Samples: {len(self.X)}")
        print(f"  Features: {self.X.shape[1]}")
        print(f"  Default target: {self._target_column}")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        Load and return YAML configuration.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Dictionary with configuration settings.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @property
    def y(self) -> pd.Series:
        """Get the current target variable."""
        return self.metadata[self._target_column]

    @property
    def target_column(self) -> str:
        """Get the current target column name."""
        return self._target_column

    def set_target(self, target_column: str) -> pd.Series:
        """
        Switch to a different target variable.

        Args:
            target_column: Name of the target column in metadata.

        Returns:
            Series with the new target values.
        """
        if target_column not in self.metadata.columns:
            available = ', '.join(self.metadata.columns)
            raise ValueError(f"Target '{target_column}' not found. Available: {available}")

        self._target_column = target_column
        print(f"Target switched to: {target_column}")
        return self.y

    def get_available_targets(self) -> list:
        """Get list of available target columns."""
        return self.metadata.columns.tolist()

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a single sample or slice of samples.

        Args:
            idx: Integer index, slice, or list of indices.

        Returns:
            Tuple of (features, target) for the sample(s).
        """
        if isinstance(idx, (int, slice, list)):
            return self.X.iloc[idx], self.y.iloc[idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def get_target(self, target_column: str) -> pd.Series:
        """
        Get a specific target without switching the current target.

        Args:
            target_column: Name of the target column in metadata.

        Returns:
            Series with target values.
        """
        if target_column not in self.metadata.columns:
            available = ', '.join(self.metadata.columns)
            raise ValueError(f"Target '{target_column}' not found. Available: {available}")
        return self.metadata[target_column]

    def _load_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Internal method to load and clean dataset."""
        report = {'dataset_name': self.config['name']}

        # Resolve paths relative to config file location
        config_dir = Path(self.config_path).parent.parent
        features_path = config_dir / self.config['paths']['features']
        metadata_path = config_dir / self.config['paths']['metadata']

        # Load features
        separator = self.config.get('separator', ',')
        features = pd.read_csv(features_path, sep=separator, index_col=0)

        # Transpose if needed
        if self.config.get('transpose_features', False):
            features = features.T

        report['initial_samples'] = features.shape[0]
        report['initial_features'] = features.shape[1]

        # Load metadata
        metadata = pd.read_csv(metadata_path, sep=separator, index_col=0)

        # Set sample ID as index
        sample_id_col = self.config['sample_id_column']
        if sample_id_col in metadata.columns:
            metadata = metadata.set_index(sample_id_col)

        # Align features and metadata
        common_samples = features.index.intersection(metadata.index)

        report['samples_not_in_metadata'] = len(features.index) - len(common_samples)
        report['samples_not_in_features'] = len(metadata.index) - len(common_samples)

        X = features.loc[common_samples]
        metadata = metadata.loc[common_samples]

        # Clean data
        X, metadata, clean_report = self._clean_data(X, metadata)
        report.update(clean_report)

        report['final_samples'] = X.shape[0]
        report['final_features'] = X.shape[1]
        report['target_column'] = self.config['target_column']
        report['missing_values_remaining'] = X.isnull().sum().sum()

        return X, metadata, pd.Series(report)

    @staticmethod
    def _clean_data(X: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Clean feature matrix by removing invalid rows and columns.

        Removes:
        - Rows (samples) that are completely NaN
        - Columns (features) that are completely NaN
        - Columns (features) with only one unique value (excluding NaN)

        Args:
            X: Feature matrix (samples × features)
            metadata: Sample metadata

        Returns:
            Tuple of (cleaned X, cleaned metadata, report dict)
        """
        report = {}

        # Remove rows that are completely NaN
        rows_all_nan = X.isnull().all(axis=1)
        report['samples_removed_all_nan'] = int(rows_all_nan.sum())
        if rows_all_nan.any():
            rows_to_keep = rows_all_nan == False  # noqa: E712
            X = X.loc[rows_to_keep]
            metadata = metadata.loc[rows_to_keep]

        # Remove columns that are completely NaN
        cols_all_nan = X.isnull().all(axis=0)
        report['features_removed_all_nan'] = int(cols_all_nan.sum())
        if cols_all_nan.any():
            cols_to_keep = cols_all_nan == False  # noqa: E712
            X = X.loc[:, cols_to_keep]

        # Remove columns with only one unique value (excluding NaN)
        nunique = X.nunique(dropna=True)
        cols_single_value = nunique <= 1
        report['features_removed_single_value'] = int(cols_single_value.sum())
        if cols_single_value.any():
            cols_to_keep = cols_single_value == False  # noqa: E712
            X = X.loc[:, cols_to_keep]

        return X, metadata, report


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <config_path>")
        print("Example: python data_loader.py configs/scanb.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load using DataLoader class
    loader = DataLoader(config_path)

    print(f"\n=== Loading Report ===")
    print(loader.report.to_string())
    print(f"\nAvailable targets: {loader.get_available_targets()}")
    print(f"Current target: {loader.target_column}")
    print(f"\nFirst 5 samples:")
    print(loader.X.head())
