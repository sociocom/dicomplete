import sys

sys.path.append("src")

from src.main import main
from src.generate import Trainer, GenerateText
import warnings

warnings.filterwarnings("ignore")


def test_main():
    main(
        fname="data/inputs/sample/disease_semple100.csv",
        input_column="出現形",
        predict_column="正規形",
        reliability_column="正規形チェックフラグ",
        rank="C",
        batch_size=4,
        epoch_num=1,
        device="mps",
    )
    assert True
