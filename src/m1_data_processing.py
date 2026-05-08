"""
src/m1_data_processing.py
M1 数据处理模块：加载 → 质量报告 → 清洗 → 特征工程。
作业要求：缺失率/异常值统计、清洗策略注释、时间特征、≥2 个衍生特征。
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ====== 路径常量 ======
ROOT = Path(__file__).parent.parent          # 即 taxi_project/
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
RAW_FILE = DATA_DIR / "yellow_tripdata_2023-01.parquet"
ZONE_FILE = DATA_DIR / "taxi_zone_lookup.csv"
CLEANED_FILE = DATA_DIR / "yellow_2023-01_cleaned.parquet"

# 纽约三大机场对应的 LocationID（从 taxi_zone_lookup 中可查）
AIRPORT_ZONES = {1: "Newark (EWR)", 132: "JFK", 138: "LGA"}


# ====== 加载 ======
def load_raw_data() -> pd.DataFrame:
    print(f"加载原始数据: {RAW_FILE.name}")
    df = pd.read_parquet(RAW_FILE)
    print(f"  原始记录数: {len(df):,}")
    print(f"  字段:       {list(df.columns)}")
    return df


def load_zone_lookup() -> pd.DataFrame:
    return pd.read_csv(ZONE_FILE)


# ====== 数据质量报告 ======
def generate_quality_report(df: pd.DataFrame, save: bool = True) -> dict:
    """统计缺失率、异常值，并保存为 markdown 报告。"""
    report = {"total_records": len(df)}

    # 缺失值
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(3)
    miss_df = pd.DataFrame({
        "字段": missing.index,
        "缺失数量": missing.values,
        "缺失率(%)": missing_pct.values,
    })
    miss_df = miss_df[miss_df["缺失数量"] > 0]
    report["missing"] = miss_df

    # 异常值
    outliers = {}
    outliers["车费 ≤ 0"] = int((df["fare_amount"] <= 0).sum())
    outliers["车费 > 500"] = int((df["fare_amount"] > 500).sum())
    outliers["距离 = 0"] = int((df["trip_distance"] == 0).sum())
    outliers["距离 > 100 英里"] = int((df["trip_distance"] > 100).sum())
    outliers["乘客数 = 0 或 NaN"] = int(
        ((df["passenger_count"].isna()) | (df["passenger_count"] == 0)).sum()
    )
    outliers["乘客数 > 6"] = int((df["passenger_count"] > 6).sum())

    duration_sec = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    outliers["行程时长 ≤ 0"] = int((duration_sec <= 0).sum())
    outliers["行程时长 > 4 小时"] = int((duration_sec > 4 * 3600).sum())
    outliers["不在 2023年1月"] = int(
        ((df["tpep_pickup_datetime"] < pd.Timestamp("2023-01-01")) |
         (df["tpep_pickup_datetime"] >= pd.Timestamp("2023-02-01"))).sum()
    )
    report["outliers"] = outliers

    # 控制台打印
    print("\n" + "=" * 60)
    print("数据质量报告")
    print("=" * 60)
    print(f"总记录数: {report['total_records']:,}")
    print("\n【缺失值】")
    print("  无缺失" if miss_df.empty else miss_df.to_string(index=False))
    print("\n【异常值】")
    for k, v in outliers.items():
        print(f"  {k:<22}{v:>10,}  ({v/len(df)*100:.2f}%)")

    # 保存 markdown
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        md_path = OUTPUTS_DIR / "m1_data_quality_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# M1 数据质量报告\n\n")
            f.write(f"**数据文件**: `{RAW_FILE.name}`  \n")
            f.write(f"**总记录数**: {report['total_records']:,}\n\n")
            f.write("## 一、缺失值\n\n")
            if miss_df.empty:
                f.write("无缺失值。\n\n")
            else:
                f.write("| 字段 | 缺失数量 | 缺失率(%) |\n|---|---:|---:|\n")
                for _, row in miss_df.iterrows():
                    f.write(f"| {row['字段']} | {row['缺失数量']:,} | {row['缺失率(%)']:.3f} |\n")
                f.write("\n")
            f.write("## 二、异常值统计\n\n")
            f.write("| 异常情况 | 记录数 | 占比 |\n|---|---:|---:|\n")
            for k, v in outliers.items():
                f.write(f"| {k} | {v:,} | {v/len(df)*100:.2f}% |\n")
        print(f"\n  报告已保存: {md_path.relative_to(ROOT)}")

    return report


# ====== 清洗 ======
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    分 6 步清洗，每步都说明理由（作业明确要求"在注释中说明每步策略的理由"）。
    """
    print("\n" + "=" * 60)
    print("数据清洗")
    print("=" * 60)
    n0 = len(df)
    print(f"清洗前: {n0:,} 条")

    # [1] 删除关键时间字段缺失的记录
    # 理由: pickup/dropoff 时间是后续所有时间分析的基础，缺失即无价值
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    print(f"  [1] 删除时间缺失:        {n0 - len(df):>9,} 条")

    # [2] 仅保留 2023 年 1 月内的行程
    # 理由: 数据集本应只含 2023-01，但实际混入了少量 2022 年末和 2023 年 2 月的记录
    n1 = len(df)
    df = df[(df["tpep_pickup_datetime"] >= pd.Timestamp("2023-01-01")) &
            (df["tpep_pickup_datetime"] < pd.Timestamp("2023-02-01"))]
    print(f"  [2] 限定 2023年1月:       {n1 - len(df):>9,} 条")

    # [3] 计算行程时长，过滤 0~240 分钟之外的记录
    # 理由: 时长≤0 是录入错误；>4小时基本可判定为计价器忘关，不能反映真实出行需求
    df = df.copy()
    df["trip_duration_min"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    )
    n2 = len(df)
    df = df[(df["trip_duration_min"] > 0) & (df["trip_duration_min"] <= 240)]
    print(f"  [3] 时长合理 (0,240]分钟: {n2 - len(df):>9,} 条")

    # [4] 行程距离 > 0 且 ≤ 100 英里
    # 理由: 距离=0 多半是取消订单；>100英里 已超出纽约市辖范围，应剔除
    n3 = len(df)
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]
    print(f"  [4] 距离合理 (0,100]英里: {n3 - len(df):>9,} 条")

    # [5] 车费 > 0 且 ≤ 500 美元
    # 理由: 负车费 = 退款记录（不计入需求分析）；>500 美元 = 极端异常或欺诈
    n4 = len(df)
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 500)]
    print(f"  [5] 车费合理 (0,500]美元: {n4 - len(df):>9,} 条")

    # [6] 乘客数 1~6
    # 理由: 出租车物理上限 6 人；0 人或缺失是设备故障，不是真实出行
    n5 = len(df)
    df = df[df["passenger_count"].between(1, 6)]
    print(f"  [6] 乘客数 1-6:           {n5 - len(df):>9,} 条")

    print(f"清洗后: {len(df):,} 条 (保留率 {len(df)/n0*100:.2f}%)")
    return df.reset_index(drop=True)


# ====== 时间特征 ======
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 pickup 时间提取小时、星期、是否周末、是否高峰。"""
    df = df.copy()
    pickup = df["tpep_pickup_datetime"]
    df["hour"] = pickup.dt.hour
    df["day_of_week"] = pickup.dt.dayofweek      # 0=周一 … 6=周日
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # 高峰: 工作日 7-9 点 或 17-19 点
    df["is_peak"] = (
        (df["is_weekend"] == 0) &
        (df["hour"].between(7, 9) | df["hour"].between(17, 19))
    ).astype(int)
    df["date"] = pickup.dt.date
    return df


# ====== 衍生特征 ======
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    4 个自行设计的衍生特征（作业要求至少 2 个，多做不扣分）：
      1. trip_speed_mph   - 平均速度，反映路况/拥堵
      2. fare_per_mile    - 每英里单价，识别短途起步价偏高 / 异常计价
      3. tip_ratio        - 小费占车费比，反映服务质量与乘客类型
      4. is_airport_trip  - 是否涉及三大机场，机场行程通常更长更贵，是单独的市场
    """
    df = df.copy()
    df["trip_speed_mph"] = df["trip_distance"] / (df["trip_duration_min"] / 60)
    df["trip_speed_mph"] = df["trip_speed_mph"].replace([np.inf, -np.inf], np.nan)

    df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"]
    df["fare_per_mile"] = df["fare_per_mile"].replace([np.inf, -np.inf], np.nan)

    df["tip_ratio"] = df.get("tip_amount", 0) / df["fare_amount"]

    df["is_airport_trip"] = (
        df["PULocationID"].isin(AIRPORT_ZONES) |
        df["DOLocationID"].isin(AIRPORT_ZONES)
    ).astype(int)

    return df


# ====== 主流程 ======
def process_pipeline(use_cache: bool = True) -> pd.DataFrame:
    """
    M1 主入口。use_cache=True 时若已有清洗结果就跳过重跑，
    方便 M2/M3/M4 调试时秒速加载。
    """
    if use_cache and CLEANED_FILE.exists():
        print(f"读取缓存的清洗后数据: {CLEANED_FILE.name}")
        return pd.read_parquet(CLEANED_FILE)

    df_raw = load_raw_data()
    generate_quality_report(df_raw, save=True)
    df_clean = clean_data(df_raw)
    df = add_time_features(df_clean)
    df = add_derived_features(df)

    df.to_parquet(CLEANED_FILE, index=False)
    print(f"\n清洗后数据已缓存: {CLEANED_FILE.name}  ({CLEANED_FILE.stat().st_size/1024/1024:.1f} MB)")
    print(f"最终字段: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = process_pipeline(use_cache=False)
    print(f"\n最终数据形状: {df.shape}")
    print("\n前 3 行示例:")
    print(df.head(3))