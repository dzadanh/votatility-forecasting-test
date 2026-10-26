from datetime import date, datetime
from pathlib import Path

from vnstock import Listing, Quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download_company_data(company_name: str, start_date: str, end_date: str | None = None):
    """Tải dữ liệu giá theo ngày cho công ty và lưu thành CSV.

    company_name có thể là mã chứng khoán (ví dụ ACB) hoặc tên công ty.
    Ngày nhập theo định dạng YYYY-MM-DD. Bỏ trống end_date để lấy đến hôm nay.
    """
    end_date = end_date or date.today().isoformat()
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("Ngày bắt đầu phải trước hoặc bằng ngày kết thúc.")

    # Tra tên công ty thành mã giao dịch (hoặc chấp nhận nhập mã trực tiếp).
    matches = Listing(source="VCI").search_symbol(company_name)
    if matches is None or matches.empty:
        raise ValueError(f"Không tìm thấy công ty hoặc mã chứng khoán: {company_name}")

    query = company_name.strip().casefold()
    exact_symbol = matches[matches["symbol"].astype(str).str.casefold() == query]
    if not exact_symbol.empty:
        match = exact_symbol.iloc[0]
    else:
        name_matches = matches[
            matches["organ_name"].astype(str).str.casefold().str.contains(query, regex=False)
        ]
        if len(name_matches) == 1:
            match = name_matches.iloc[0]
        elif len(matches) == 1:
            match = matches.iloc[0]
        else:
            choices = ", ".join(
                f"{row.symbol} ({row.organ_name})" for row in matches.itertuples()
            )
            raise ValueError(f"Tên tìm được nhiều kết quả; hãy nhập mã chứng khoán. Kết quả: {choices}")

    symbol = str(match["symbol"]).upper()
    data = Quote(symbol=symbol, source="VCI").history(
        start=start.isoformat(), end=end.isoformat(), interval="1D"
    )
    if data.empty:
        raise RuntimeError(f"Không có dữ liệu cho {symbol} trong khoảng ngày đã chọn.")

    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{symbol.lower()}.csv"
    data.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Đã lưu {len(data)} dòng dữ liệu của {symbol} vào: {output}")
    return data


if __name__ == "__main__":
    # Có thể truyền mã "ACB" hoặc tên đầy đủ, ví dụ "Ngân hàng Thương mại Cổ phần Á Châu".
    ##download_company_data("ACB", start_date="2018-01-01")
    ##download_company_data("FPT", start_date="2018-01-01")
    ##download_company_data("HPG", start_date="2003-01-01", end_date="2009-12-31")
      # Ví dụ ngày kết thúc trước ngày bắt đầu để kiểm tra lỗi
    download_company_data("HPG", start_date="2020-01-01")
    
