"""
src/m2_visualization.py
M2 分析可视化：4 项分析，共 7 张图，全部自动保存至 outputs/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ====== 路径 ======
ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"
ZONE_FILE = ROOT / "data" / "taxi_zone_lookup.csv"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ====== 全局图表样式（支持中文） ======
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_theme(style="whitegrid")


def load_zone_lookup() -> dict:
    """加载 LocationID -> 区域名称 映射"""
    zones = pd.read_csv(ZONE_FILE)
    return dict(zip(zones["LocationID"], zones["Zone"]))


# ====== M2.1 出行需求时间规律 ======
def plot_hourly_demand(df: pd.DataFrame) -> None:
    """分小时平均订单量折线图：工作日双峰 vs 周末平滑曲线"""
    # 用每类型的"天数"做分母，计算各小时【平均】订单量
    day_counts = df.groupby("is_weekend")["date"].nunique()
    hourly = df.groupby(["is_weekend", "hour"]).size().reset_index(name="trips")
    hourly["avg_trips"] = hourly.apply(
        lambda r: r["trips"] / day_counts.get(r["is_weekend"], 1), axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, group in hourly.groupby("is_weekend"):
        name, color = ("周末", "#E74C3C") if label else ("工作日", "#2E86AB")
        ax.plot(group["hour"], group["avg_trips"], marker="o", markersize=5,
                linewidth=2, label=name, color=color)

    ax.set_title("分小时平均出租车订单量（工作日 vs 周末）", fontsize=14, fontweight="bold")
    ax.set_xlabel("小时（0-23时）", fontsize=12)
    ax.set_ylabel("平均订单量（单/小时）", fontsize=12)
    ax.set_xticks(range(24))
    ax.legend(fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save(fig, "m2_1_hourly_demand.png")


def plot_weekday_demand(df: pd.DataFrame) -> None:
    """分星期日均订单量柱状图"""
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    daily = df.groupby(["day_of_week", "date"]).size().reset_index(name="trips")
    weekday_avg = daily.groupby("day_of_week")["trips"].mean()

    colors = ["#2E86AB"] * 5 + ["#E74C3C"] * 2
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([weekday_names[i] for i in weekday_avg.index],
                  weekday_avg.values, color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=10)

    ax.set_title("各星期日均订单量（蓝=工作日，红=周末）", fontsize=14, fontweight="bold")
    ax.set_xlabel("星期", fontsize=12)
    ax.set_ylabel("日均订单量", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    _save(fig, "m2_1_weekday_demand.png")


# ====== M2.2 区域热度分析 ======
def plot_top10_zones(df: pd.DataFrame, zone_map: dict) -> None:
    """TOP 10 上下客区域并排柱状图"""
    top_pu = df["PULocationID"].value_counts().head(10)
    top_do = df["DOLocationID"].value_counts().head(10)
    top_pu.index = [zone_map.get(i, f"Zone {i}") for i in top_pu.index]
    top_do.index = [zone_map.get(i, f"Zone {i}") for i in top_do.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(x=top_pu.values, y=top_pu.index, ax=ax1, palette="Blues_r")
    ax1.set_title("TOP 10 上客区域", fontsize=13, fontweight="bold")
    ax1.set_xlabel("上客次数", fontsize=11)
    for i, v in enumerate(top_pu.values):
        ax1.text(v + 200, i, f"{v:,}", va="center", fontsize=9)

    sns.barplot(x=top_do.values, y=top_do.index, ax=ax2, palette="Reds_r")
    ax2.set_title("TOP 10 下客区域", fontsize=13, fontweight="bold")
    ax2.set_xlabel("下客次数", fontsize=11)
    for i, v in enumerate(top_do.values):
        ax2.text(v + 200, i, f"{v:,}", va="center", fontsize=9)

    plt.suptitle("纽约出租车区域热度分析（2023年1月）", fontsize=15, fontweight="bold", y=1.02)
    _save(fig, "m2_2_top10_zones.png")


def plot_peak_heatmap(df: pd.DataFrame) -> None:
    """星期 × 小时 出行量热力图"""
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    pivot = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    pivot.index = weekday_names

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "出行次数"})
    ax.set_title("出行量热力图（星期 × 小时）", fontsize=14, fontweight="bold")
    ax.set_xlabel("小时（0-23时）", fontsize=12)
    ax.set_ylabel("星期", fontsize=12)
    _save(fig, "m2_2_heatmap.png")


# ====== M2.3 车费影响因素分析 ======
def plot_fare_analysis(df: pd.DataFrame) -> None:
    """距离-车费散点图（10万抽样）+ 各时段平均车费折线图"""
    # 图1: 散点图（必须抽样，300万点无法渲染）
    sample = df.sample(n=min(100_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["trip_distance"], sample["fare_amount"],
               alpha=0.05, s=5, color="#2E86AB")
    ax.set_title("行程距离 vs 车费（10万条抽样）", fontsize=14, fontweight="bold")
    ax.set_xlabel("行程距离（英里）", fontsize=12)
    ax.set_ylabel("车费（美元）", fontsize=12)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 150)
    _save(fig, "m2_3_distance_fare_scatter.png")

    # 图2: 各时段平均车费
    fare_hour = df.groupby(["is_weekend", "hour"])["fare_amount"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, group in fare_hour.groupby("is_weekend"):
        name, color = ("周末", "#E74C3C") if label else ("工作日", "#2E86AB")
        ax.plot(group["hour"], group["fare_amount"], marker="o", markersize=5,
                linewidth=2, label=name, color=color)
    ax.set_title("各时段平均车费（工作日 vs 周末）", fontsize=14, fontweight="bold")
    ax.set_xlabel("小时（0-23时）", fontsize=12)
    ax.set_ylabel("平均车费（美元）", fontsize=12)
    ax.set_xticks(range(24))
    ax.legend(fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}"))
    _save(fig, "m2_3_fare_by_hour.png")


# ====== M2.4 自选分析：拥堵指数 ======
def plot_congestion_index(df: pd.DataFrame) -> None:
    """
    各时段行程中位速度（拥堵指数）
    洞察价值: 速度最低的时段 = 路最堵；与需求曲线叠合可揭示"堵且难打到车"的双重压力时段。
    方法: 过滤 1-60 mph 范围内的合理速度再取中位数（避免极端值拉偏）
    """
    valid = df[df["trip_speed_mph"].between(1, 60)]
    speed = valid.groupby(["is_weekend", "hour"])["trip_speed_mph"].median().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, group in speed.groupby("is_weekend"):
        name, color = ("周末", "#E74C3C") if label else ("工作日", "#2E86AB")
        ax.plot(group["hour"], group["trip_speed_mph"], marker="s", markersize=5,
                linewidth=2, label=name, color=color)

    ax.axvspan(7, 9, alpha=0.12, color="orange", label="早高峰区间")
    ax.axvspan(17, 19, alpha=0.12, color="purple", label="晚高峰区间")
    ax.set_title("各时段行程中位速度（拥堵指数，速度越低越堵）", fontsize=14, fontweight="bold")
    ax.set_xlabel("小时（0-23时）", fontsize=12)
    ax.set_ylabel("中位速度（英里/小时）", fontsize=12)
    ax.set_xticks(range(24))
    ax.legend(fontsize=11)
    _save(fig, "m2_4_congestion_speed.png")


# ====== 工具函数 ======
def _save(fig, filename: str) -> None:
    out = OUTPUTS_DIR / filename
    fig.savefig(out)
    plt.close(fig)
    print(f"  已保存: {filename}")


# ====== 主流程 ======
def run_all_visualizations(df: pd.DataFrame) -> None:
    zone_map = load_zone_lookup()
    print("\n【M2.1】出行需求时间规律")
    plot_hourly_demand(df)
    plot_weekday_demand(df)
    print("\n【M2.2】区域热度分析")
    plot_top10_zones(df, zone_map)
    plot_peak_heatmap(df)
    print("\n【M2.3】车费影响因素分析")
    plot_fare_analysis(df)
    print("\n【M2.4】自选分析：拥堵指数")
    plot_congestion_index(df)
    print(f"\nM2 完成！7 张图已保存至 outputs/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from m1_data_processing import process_pipeline
    df = process_pipeline(use_cache=True)
    run_all_visualizations(df)