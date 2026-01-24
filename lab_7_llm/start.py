"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
from unittest import result
from core_utils.llm.time_decorator import report_time
from lab_7_llm.main import RawDataImporter
from lab_7_llm.main import RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    importer = RawDataImporter('papluca/language-identification')
    importer.obtain()
    result = importer.raw_data
    assert result is not None, "Demo does not work correctly"

    print(result.head())

    preprocessor = RawDataPreprocessor(importer.raw_data)
    analysis = preprocessor.analyze()

    print('Dataset analysis results:')
    for key, value in analysis.items():
        print(f'{key}: {value}')


if __name__ == "__main__":
    main()
