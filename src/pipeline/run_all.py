"""Run data preparation, pre-match feature engineering, training and evaluation."""
from src.data.prepare_data import prepare_data
from src.features.build_features import build_features
from src.models.train_models import train_and_evaluate


def main() -> None:
    print("\n[1/3] PREPARE DATA")
    prepare_data()
    print("\n[2/3] BUILD PRE-MATCH FEATURES")
    build_features()
    print("\n[3/3] TRAIN AND EVALUATE")
    train_and_evaluate()
    print("\nHOÀN TẤT. Xem reports/ và models/artifacts/ để kiểm tra đầu ra.")


if __name__ == "__main__":
    main()
