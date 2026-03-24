# nyiso_eia.py
import io
import os
from pathlib import Path

import requests
import numpy as np
import pandas as pd

DEFAULT_URL = "https://www.eia.gov/electricity/wholesalemarkets/csv/nyiso_load_act_hr_2024.csv"


def _detect_header_row(file_content: str, max_lines: int = 20) -> int:
    lines = file_content.splitlines()
    header_row_index = 0
    for i, line in enumerate(lines[:max_lines]):
        if ("Time" in line) or ("Period" in line) or ("Date" in line):
            header_row_index = i
            break
    return header_row_index


def load_eia_csv_with_auto_header(url: str) -> pd.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()
    file_content = resp.text

    header_row_index = _detect_header_row(file_content)
    df = pd.read_csv(io.StringIO(file_content), header=header_row_index)
    df.columns = df.columns.str.strip()
    return df


def make_nyiso_power_temp_solar(
    url: str,
    start: str = "2024-07-01",
    end: str = "2024-07-07",
    out_dir: str = ".",
    encoding: str = "cp949",
    seed: int | None = 42,
):
    if seed is not None:
        np.random.seed(seed)

    df_eia = load_eia_csv_with_auto_header(url)

    target_date_cols = [
        c for c in df_eia.columns
        if ("time" in c.lower()) or ("period" in c.lower()) or ("date" in c.lower())
    ]
    target_load_cols = [
        c for c in df_eia.columns
        if ("load" in c.lower()) or ("value" in c.lower())
    ]

    if not target_date_cols or not target_load_cols:
        raise ValueError(f"날짜 또는 부하 컬럼 감지 실패. 현재 컬럼: {list(df_eia.columns)}")

    date_col = target_date_cols[0]
    load_col = target_load_cols[0]

    df_eia[date_col] = pd.to_datetime(df_eia[date_col])
    df_eia = df_eia.sort_values(date_col)

    # 첨부 파일처럼 2024-07-07 00:00까지만 포함되게 하려면 현재 방식 유지
    mask = (df_eia[date_col] >= start) & (df_eia[date_col] <= end)
    df_sample = df_eia.loc[mask].copy()

    if df_sample.empty:
        raise ValueError(
            f"{start}~{end} 구간 데이터가 비었습니다. 범위: "
            f"{df_eia[date_col].min()} ~ {df_eia[date_col].max()}"
        )

    df_sample["Hour"] = df_sample[date_col].dt.hour
    dates = pd.to_datetime(df_sample[date_col].values)
    n = len(dates)

    # 지역별 전력 사용량 분배
    df_sample["NYC"] = df_sample[load_col] * 0.45
    df_sample["NJ"]  = df_sample[load_col] * 0.30
    df_sample["CT"]  = df_sample[load_col] * 0.25

    # 기온 생성
    # 기존 코드보다 오후에 높아지도록 수정
    hour = df_sample["Hour"].to_numpy()
    temp_data = 28 + 5 * np.cos((hour - 13) * 2 * np.pi / 24) + np.random.normal(0, 1, n)

    # 태양광 생성
    solar_gen = np.zeros(n)
    for i in range(n):
        h = int(hour[i])
        if 6 <= h <= 19:
            eff = np.exp(-((h - 13) ** 2) / 6)
            solar_gen[i] = eff * 90 * np.random.choice([1, 0.6], p=[0.8, 0.2])

    # 첨부 파일과 같은 단일 CSV 형식으로 합치기
    final_df = pd.DataFrame({
        "NYC": df_sample["NYC"].round(6),
        "NJ": df_sample["NJ"].round(6),
        "CT": df_sample["CT"].round(6),
        "datetime": dates.strftime("%Y-%m-%d %-H:00") if os.name != "nt" else dates.strftime("%Y-%m-%d %#H:00"),
        "solar": np.round(solar_gen, 1),
        "temp_c": np.round(temp_data, 1),
    })

    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(f"{out_dir}/실습데이터.csv", index=False, encoding=encoding)

    return final_df


def plot_power(final_df, n_hours: int = 48):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(final_df.index[:n_hours], final_df["NYC"][:n_hours], label="NYC", marker="o")
    plt.title("2024년 7월 전력 수요 패턴 (EIA 실제 데이터)")
    plt.legend()
    plt.grid(True)
    plt.show()


def _default_out_dir() -> str:
    if "__file__" in globals():
        return str(Path(__file__).resolve().parent)
    return os.getcwd()


def run_on_import(
    url: str = DEFAULT_URL,
    out_dir: str | None = None,
    start: str = "2024-07-01",
    end: str = "2024-07-07",
    encoding: str = "cp949",
    seed: int | None = 42,
):
    if out_dir is None:
        out_dir = _default_out_dir()
    return make_nyiso_power_temp_solar(
        url=url, start=start, end=end, out_dir=out_dir, encoding=encoding, seed=seed
    )


_AUTORUN = os.environ.get("NYISO_EIA_AUTORUN", "1") == "1"

if _AUTORUN and not globals().get("_NYISO_EIA_ALREADY_RAN", False):
    globals()["_NYISO_EIA_ALREADY_RAN"] = True
    out_dir_used = _default_out_dir()
    try:
        run_on_import(out_dir=out_dir_used)
        print(f"✅ nyiso_eia import 완료: CSV 생성됨 (저장폴더: {out_dir_used})")
    except Exception as e:
        print(f"⚠️ nyiso_eia import 자동 실행 실패: {e}")
