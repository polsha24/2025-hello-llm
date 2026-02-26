"""
Laboratory work.

Fine-tuning Large Language Models for a downstream task.
"""

# pylint: disable=too-few-public-methods, undefined-variable, duplicate-code, unused-argument, too-many-arguments
from pathlib import Path
from typing import Callable, Iterable, Sequence

import evaluate
import pandas as pd
from datasets import load_dataset
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time

try:
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    print('Library "transformers" not installed. Failed to import.')

try:
    from torchinfo import summary  # type: ignore
except ImportError:
    print('Library "torchinfo" not installed. Failed to import.')

try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')


class RawDataImporter(AbstractRawDataImporter):
    """
    Custom implementation of data importer.
    """

    @report_time
    def obtain(self) -> None:
        """
        Import dataset.
        """
        self._raw_data = load_dataset(self._hf_name, split="validation").to_pandas()
        if not isinstance(self._raw_data, pd.DataFrame):
            raise TypeError("Downloaded dataset is not a pandas DataFrame")


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    Custom implementation of data preprocessor.
    """

    def analyze(self) -> dict:
        """
        Analyze preprocessed dataset.

        Returns:
            dict: dataset key properties.
        """
        if "text" in self._raw_data.columns:
            series = self._raw_data["text"].dropna().astype(str)
        elif "translation" in self._raw_data.columns:
            def get_source(translation: object) -> str | None:
                if isinstance(translation, dict):
                    if "ru" in translation:
                        return cast(str, translation["ru"])
                    if translation:
                        return cast(str, next(iter(translation.values())))
                return None

            series = self._raw_data["translation"].map(get_source).dropna().astype(str)
        elif {"ru", "es"}.issubset(self._raw_data.columns):
            series = self._raw_data["ru"].dropna().astype(str)
        else:
            series = pd.Series(dtype=str)

        lengths = series.str.len()

        return {
            "dataset_number_of_samples": len(self._raw_data),
            "dataset_columns": len(self._raw_data.columns),
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isnull().all(axis=1).sum(),
            "dataset_sample_min_len": int(lengths.min()) if not lengths.empty else 0,
            "dataset_sample_max_len": int(lengths.max()) if not lengths.empty else 0,
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        if "translation" in self._raw_data.columns:

            def get_source(translation: object) -> str | None:
                if isinstance(translation, dict) and "ru" in translation:
                    return cast(str, translation["ru"])
                return None

            def get_target(translation: object) -> str | None:
                if isinstance(translation, dict) and "es" in translation:
                    return cast(str, translation["es"])
                return None

            translations = self._raw_data["translation"]
            data = pd.DataFrame(
                {
                    ColumnNames.SOURCE.value: translations.map(get_source),
                    ColumnNames.TARGET.value: translations.map(get_target),
                }
            )
        elif {"ru", "es"}.issubset(self._raw_data.columns):
            data = pd.DataFrame(
                {
                    ColumnNames.SOURCE.value: self._raw_data["ru"].astype(str),
                    ColumnNames.TARGET.value: self._raw_data["es"].astype(str),
                }
            )
        else:
            data = self._raw_data.rename(
                columns={
                    "text": ColumnNames.SOURCE.value,
                    "labels": ColumnNames.TARGET.value,
                }
            )

        self._data = data.dropna(subset=[ColumnNames.SOURCE.value, ColumnNames.TARGET.value]).reset_index(
            drop=True
        )


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        row = self._data.iloc[index]
        return (
            row[ColumnNames.SOURCE.value],
            row[ColumnNames.TARGET.value],
        )

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


def tokenize_sample(
    sample: pd.Series, tokenizer: AutoTokenizer, max_length: int
) -> dict[str, torch.Tensor]:
    """
    Tokenize sample.

    Args:
        sample (pandas.Series): sample from a dataset
        tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to tokenize
            original data
        max_length (int): max length of sequence

    Returns:
        dict[str, torch.Tensor]: Tokenized sample
    """


class TokenizedTaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
            tokenizer (transformers.models.auto.tokenization_auto.AutoTokenizer): Tokenizer to
                tokenize the dataset
            max_length (int): max length of a sequence
        """

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            dict[str, torch.Tensor]: An element from the dataset
        """


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """

    def __init__(
        self, model_name: str, dataset: TaskDataset, max_length: int, batch_size: int, device: str
    ) -> None:
        """
        Initialize an instance.

        Args:
            model_name (str): The name of the pre-trained model.
            dataset (TaskDataset): The dataset to be used for translation.
            max_length (int): The maximum length of generated sequence.
            batch_size (int): The size of the batch inside DataLoader.
            device (str): The device for inference.
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self._device)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """
        assert self._model is not None
        config = self._model.config

        module = cast(torch.nn.Module, self._model)
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in module.parameters())

        max_context_length = int(
            getattr(
                config,
                "max_position_embeddings",
                getattr(config, "max_length", self._max_length),
            )
        )
        vocab_size = int(getattr(config, "vocab_size", 0))
        embedding_size = int(
            getattr(
                config,
                "d_model",
                getattr(config, "hidden_size", max_context_length),
            )
        )

        return {
            "input_shape": [1, max_context_length],
            "embedding_size": embedding_size,
            "output_shape": [1, max_context_length, vocab_size],
            "num_trainable_params": int(trainable_params),
            "vocab_size": vocab_size,
            "size": int(total_params * 4),
            "max_context_length": max_context_length,
        }

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if self._model is None:
            return None
        predictions = self._infer_batch([sample])
        return predictions[0] if predictions else None

    @report_time
    def infer_dataset(self) -> pd.DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        data_loader = DataLoader(
            self._dataset,
            batch_size=self._batch_size,
        )
        all_targets: list[str] = []
        all_predictions: list[str] = []
        for sources, targets in data_loader:
            sample_batch: list[tuple[str, ...]] = [(text,) for text in sources]
            batch_predictions = self._infer_batch(sample_batch)
            all_targets.extend(list(targets))
            all_predictions.extend(batch_predictions)
        return pd.DataFrame(
            {
                ColumnNames.TARGET.value: all_targets,
                ColumnNames.PREDICTION.value: all_predictions,
            }
        )

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): batch to infer the model

        Returns:
            list[str]: model predictions as strings
        """
        texts = [sample[0] for sample in sample_batch]
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        assert self._model is not None
        generated = self._model.generate(**encoded, max_length=self._max_length)
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """
        self._data_path = data_path
        self._metrics = metrics

    def run(self) -> dict:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict: A dictionary containing information about the calculated metric
        """
        data = pd.read_csv(self._data_path)
        target_series = data[ColumnNames.TARGET.value]
        prediction_series = data[ColumnNames.PREDICTION.value]

        result: dict[str, float] = {}
        for metric in self._metrics:
            metric_fn = evaluate.load(str(metric))

            if metric is Metrics.BLEU:
                references = [[str(ref)] for ref in target_series]
                predictions = [str(pred) for pred in prediction_series]
                computed = metric_fn.compute(
                    references=references,
                    predictions=predictions,
                )
            elif metric in (Metrics.F1, Metrics.PRECISION, Metrics.RECALL, Metrics.ACCURACY):
                references = [int(ref) for ref in target_series]
                predictions = [int(pred) for pred in prediction_series]
                compute_kwargs: dict = {
                    "references": references,
                    "predictions": predictions,
                }
                if metric in (Metrics.F1, Metrics.PRECISION, Metrics.RECALL):
                    compute_kwargs["average"] = "macro"
                computed = metric_fn.compute(**compute_kwargs)
            else:
                references = [str(ref) for ref in target_series]
                predictions = [str(pred) for pred in prediction_series]
                computed = metric_fn.compute(
                    references=references,
                    predictions=predictions,
                )

            result[str(metric)] = float(computed[str(metric)])
        return result


# class SFTPipeline(AbstractSFTPipeline):
#     """
#     A class that initializes a model, fine-tuning.
#     """

#     def __init__(
#         self,
#         model_name: str,
#         dataset: Dataset,
#         sft_params: SFTParams,
#         data_collator: Callable[[AutoTokenizer], torch.Tensor] | None = None,
#     ) -> None:
#         """
#         Initialize an instance of ClassificationSFTPipeline.

#         Args:
#             model_name (str): The name of the pre-trained model.
#             dataset (torch.utils.data.dataset.Dataset): The dataset used.
#             sft_params (SFTParams): Fine-Tuning parameters.
#             data_collator (Callable[[AutoTokenizer], torch.Tensor] | None, optional): processing
#                                                                     batch. Defaults to None.
#         """

#     def run(self) -> None:
#         """
#         Fine-tune model.
#         """
