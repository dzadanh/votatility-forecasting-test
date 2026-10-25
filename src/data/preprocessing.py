from pathlib import Path

import pandas as pd

from .loader import load_data
from .validator import validate_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch và chuẩn hóa OHLCV; feature engineering nằm ở src/features."""
    return validate_data(data, clean=True)


def preprocess_data(company_name: str, save: bool = True) -> pd.DataFrame:
    """Đọc raw, kiểm tra dữ liệu rồi lưu OHLCV sạch vào data/processed."""
    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError('company_name phải là mã chứng khoán không rỗng.')
    symbol = company_name.strip().casefold()
    cleaned = clean_data(load_data(company_name))
    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(PROCESSED_DATA_DIR / f'{symbol}_clean.csv', index=False, encoding='utf-8-sig')
    return cleaned
if __name__ == '__main__':
    preprocess_data('ACB', save=True)
    preprocess_data('FPT', save=True)
    preprocess_data('HPG', save=True)