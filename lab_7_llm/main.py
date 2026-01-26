"""
Laboratory work.

Working with Large Language Models.
"""

# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
import pandas as pd
from datasets import load_dataset
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        df = load_dataset(self._hf_name, split="validation")

        if not hasattr(df, "to_pandas"):
            raise TypeError("Downloaded dataset is not a pandas DataFrame")
        df = df.to_pandas()

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pandas DataFrame")
        self._raw_data = df


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        df = self._raw_data.copy()

        num_samples = len(df)
        num_columns = len(df.columns)
        num_duplicates = df.duplicated().sum()
        num_empty_rows = df.isnull().all(axis=1).sum()

        text_col = 'text'
        df_non_empty = df.dropna(subset=[text_col])

        lengths = df_non_empty[text_col].str.len()
        min_len = int(lengths.min())
        max_len = int(lengths.max())

        return {
            "dataset_number_of_samples": int(num_samples),
            "dataset_columns": int(num_columns),
            "dataset_duplicates": int(num_duplicates),
            "dataset_empty_rows": int(num_empty_rows),
            "dataset_sample_min_len": min_len,
            "dataset_sample_max_len": max_len,
        }
