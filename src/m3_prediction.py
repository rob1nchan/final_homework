"""
src/m3_prediction.py
M3 神经网络预测 + 随机森林对比
- PyTorch MLP 预测 (区域 × 日期 × 小时) 的订单数
- 8:2 训练/测试集，loss 曲线，测试集 MAE/RMSE
- 与 sklearn 随机森林对比，写出优劣分析
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ====== 路径与样式 ======
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
CLEANED_FILE = DATA_DIR / "yellow_2023-01_cleaned.parquet"
OUTPUTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
})

# ====== 超参数 ======
TOP_N_ZONES = 50
HIDDEN_DIMS = (128, 64, 32)
EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
TEST_SIZE = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ====== 1. 聚合 ======
def aggregate_demand(df: pd.DataFrame) -> pd.DataFrame:
    """把流水聚合成 (区域 × 日期 × 小时) 粒度的需求量。"""
    print("\n[1/5] 聚合为 (区域 × 日期 × 小时) 的需求量...")
    top_zones = df["PULocationID"].value_counts().head(TOP_N_ZONES).index
    df_top = df[df["PULocationID"].isin(top_zones)].copy()
    print(f"  TOP {TOP_N_ZONES} 区域覆盖原始数据: {len(df_top)/len(df)*100:.1f}%")

    agg = df_top.groupby(["PULocationID", "date", "hour"]).agg(
        trip_count=("PULocationID", "size"),
        day_of_week=("day_of_week", "first"),
        is_weekend=("is_weekend", "first"),
        is_peak=("is_peak", "first"),
    ).reset_index()
    agg["day_of_month"] = pd.to_datetime(agg["date"]).dt.day

    # 把 LocationID 映射成 0..N-1 索引（NN 标准化与 RF 树分裂都更稳定）
    zone_to_idx = {z: i for i, z in enumerate(top_zones)}
    agg["zone_idx"] = agg["PULocationID"].map(zone_to_idx)

    print(f"  聚合后样本数: {len(agg):,}")
    print(f"  目标分布: 平均 {agg['trip_count'].mean():.1f} / 中位 {agg['trip_count'].median():.0f} / 最大 {agg['trip_count'].max():,}")
    return agg


# ====== 2. 切分 + 标准化 ======
def prepare_data(agg: pd.DataFrame):
    print("\n[2/5] 8:2 训练/测试切分 + 特征标准化...")
    feat_cols = ["hour", "day_of_week", "is_weekend", "is_peak", "day_of_month", "zone_idx"]
    X = agg[feat_cols].values.astype(np.float32)
    y = agg["trip_count"].values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"  训练集: {len(X_tr):,}  |  测试集: {len(X_te):,}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    return X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s


# ====== 3. PyTorch MLP ======
class DemandMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(X_tr_s, y_tr, X_te_s, y_te):
    print("\n[3/5] 训练神经网络 (PyTorch MLP)...")
    # 用 log1p 稳定训练（trip_count 长尾右偏严重）
    y_tr_log = np.log1p(y_tr).astype(np.float32)
    y_te_log = np.log1p(y_te).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_log)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te_s), torch.from_numpy(y_te_log)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = DemandMLP(X_tr_s.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                va_loss += crit(model(xb), yb).item() * len(xb)
        va_loss /= len(test_loader.dataset)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:>3}/{EPOCHS}: train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}")

    # 测试集预测（反 log1p 回原空间）
    model.eval()
    with torch.no_grad():
        y_pred_log = model(torch.from_numpy(X_te_s)).numpy()
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    return model, history, y_pred


# ====== 4. 随机森林对比 ======
def train_rf(X_tr, y_tr, X_te):
    print("\n[4/5] 训练随机森林 (sklearn)...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr)
    return rf, rf.predict(X_te)


# ====== 5. 评估 + 出图 + 报告 ======
def evaluate(name: str, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"  {name:<10}  MAE = {mae:>7.2f}    RMSE = {rmse:>7.2f}")
    return mae, rmse


def plot_loss_curve(history, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ep = range(1, len(history["train_loss"]) + 1)
    ax.plot(ep, history["train_loss"], label="训练集 loss", linewidth=2, color="#2E86AB")
    ax.plot(ep, history["val_loss"], label="测试集 loss", linewidth=2, color="#E74C3C")
    ax.set_title("神经网络训练过程的 Loss 曲线", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss (log1p 空间)", fontsize=12)
    ax.legend(fontsize=11)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  loss 曲线已保存: {save_path.name}")


def plot_comparison(y_te, y_nn, y_rf, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    max_v = max(y_te.max(), y_nn.max(), y_rf.max())
    for ax, pred, name, color in [
        (axes[0], y_nn, "神经网络 (PyTorch MLP)", "#2E86AB"),
        (axes[1], y_rf, "随机森林 (RandomForest)", "#27AE60"),
    ]:
        ax.scatter(y_te, pred, alpha=0.3, s=8, color=color)
        ax.plot([0, max_v], [0, max_v], "--", color="red", linewidth=1, label="完美预测线 y=x")
        ax.set_xlabel("真实需求量", fontsize=12)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.legend()
    axes[0].set_ylabel("预测需求量", fontsize=12)
    plt.suptitle("模型预测 vs 真实值（测试集）", fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  预测对比图已保存: {save_path.name}")


def write_comparison_report(metrics, save_path):
    nn_mae, nn_rmse = metrics["NN"]
    rf_mae, rf_rmse = metrics["RF"]
    win_mae = "随机森林" if rf_mae < nn_mae else "神经网络"
    win_rmse = "随机森林" if rf_rmse < nn_rmse else "神经网络"
    md = f"""# M3 模型对比报告

## 任务定义
- **预测目标**: 给定 (区域 × 日期 × 小时)，预测该时段该区域的出租车订单数
- **数据范围**: TOP {TOP_N_ZONES} 个最热门上客区域，2023年1月
- **特征**: hour, day_of_week, is_weekend, is_peak, day_of_month, zone_idx (6 维)
- **训练/测试**: 8:2 随机划分，random_state={SEED}

## 实验结果

| 模型 | MAE | RMSE |
|---|---:|---:|
| 神经网络 (PyTorch MLP, 隐层 {HIDDEN_DIMS}) | {nn_mae:.2f} | {nn_rmse:.2f} |
| 随机森林 (n_estimators=100, max_depth=20) | {rf_mae:.2f} | {rf_rmse:.2f} |

按 MAE 表现更好: **{win_mae}**  
按 RMSE 表现更好: **{win_rmse}**

## 两种方法的优劣分析

### 神经网络
**优势**：
- 能学习特征间复杂的非线性交互（如周五晚 × 机场区域的特殊模式）
- 通过 log1p 变换稳定了对长尾需求的拟合
- 可扩展性强，未来加入天气/节假日等高维特征可以无缝接入

**劣势**：
- 在仅 6 维特征 + 约 3-4 万样本的小数据场景，模型容量过剩，容易过拟合
- 需要特征标准化、log 变换、调超参等多步预处理，工程量大
- CPU 训练慢（50 epoch 约 1-2 分钟）

### 随机森林
**优势**：
- 树模型天然适合表格数据，无需特征缩放
- 对异常值/长尾分布鲁棒
- 训练快，结果稳定，可解释（可输出特征重要性）

**劣势**：
- 难以外推（预测值受训练集最大值约束）
- 不容易学习连续型特征之间的精细关系
- 模型体积较大（100 棵树）

## 结论
对于本任务（低维结构化特征 + 中等规模数据），随机森林通常已经足够强；
神经网络的额外建模能力没有充分发挥。但作业里的"对比"本身就是教学目标 ——
让我们直观看到：**不是所有问题都需要深度学习。**
"""
    save_path.write_text(md, encoding="utf-8")
    print(f"  对比报告已保存: {save_path.name}")


# ====== 主流程 ======
def run_m3(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("M3 神经网络预测 + 随机森林对比")
    print("=" * 60)
    agg = aggregate_demand(df)
    X_tr, X_te, y_tr, y_te, X_tr_s, X_te_s = prepare_data(agg)

    nn_model, history, y_pred_nn = train_nn(X_tr_s, y_tr, X_te_s, y_te)
    rf_model, y_pred_rf = train_rf(X_tr, y_tr, X_te)

    print("\n[5/5] 测试集评估")
    metrics = {
        "NN": evaluate("神经网络", y_te, y_pred_nn),
        "RF": evaluate("随机森林", y_te, y_pred_rf),
    }

    plot_loss_curve(history, OUTPUTS_DIR / "m3_loss_curve.png")
    plot_comparison(y_te, y_pred_nn, y_pred_rf, OUTPUTS_DIR / "m3_pred_vs_truth.png")
    write_comparison_report(metrics, OUTPUTS_DIR / "m3_model_comparison.md")
    print(f"\nM3 完成！产出已保存至 outputs/")
    return nn_model, rf_model, metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from m1_data_processing import process_pipeline
    df = process_pipeline(use_cache=True)
    run_m3(df)