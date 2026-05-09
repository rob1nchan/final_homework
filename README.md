# 城市出租车出行数据分析与智能问答系统

> 《人工智能编程语言》课程期末大作业 · 周婧扬 / 25361053

基于纽约市黄色出租车 2023 年 1 月公开数据（约 300 万条）构建的端到端数据科学项目，包含数据清洗、可视化分析、神经网络预测和自然语言问答接口。

---

## 项目结构

```
taxi_project/
├── data/                            # 数据文件夹（运行 download_data.py 自动下载）
│   └── taxi_zone_lookup.csv         # 区域 ID 映射表
├── outputs/                         # 所有生成的图表和报告
├── src/
│   ├── m1_data_processing.py        # M1 数据加载、清洗、特征工程
│   ├── m2_visualization.py          # M2 7 张分析图表
│   ├── m3_prediction.py             # M3 PyTorch 神经网络 + 随机森林对比
│   ├── m4_qa_system.py              # M4 命令行问答（规则 + LLM 兜底）
│   └── secrets_config.py            # API Key 配置（已 gitignore，不会上传）
├── main.py                          # 一键运行入口
├── download_data.py                 # 自动下载数据集（首次运行必跑）
├── requirements.txt                 # Python 依赖
├── 人机协作报告.md                  # 10 分协作报告
└── README.md
```

---

## 快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
```

> Windows 用户如果 torch 安装失败，请用清华镜像 + 单独装 torch：
> ```bash
> pip install pandas numpy matplotlib seaborn scikit-learn pyarrow requests openai -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. 下载数据

```bash
python download_data.py
```

脚本会自动下载到 `data/`：
- `yellow_tripdata_2023-01.parquet`（~47 MB，约 300 万条）—— **该文件因体积超过 GitHub 单文件上限未直接入库，请用此脚本拉取**
- `taxi_zone_lookup.csv`（区域 ID 映射表，已在仓库中）

### 3. 一键跑通全流程

```bash
python main.py
```

依次执行 M1 → M2 → M3 → M4，最后进入问答循环。完整运行约 30 秒～8 分钟（取决于 M1 是否使用缓存）。

### 4.（选做加分项）启用 LLM 兜底

注册 DeepSeek（或通义千问 / 智谱 GLM）API key 后，编辑 `src/secrets_config.py`：

```python
LLM_API_KEY = "sk-你的真实key"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
```

不配置则仅启用规则问答（仍可拿基础 20 分）。

---

## 模块说明

| 模块 | 分值 | 核心交付 |
|---|---|---|
| **M1** 数据处理 | 20 | 数据质量报告 + 6 步清洗（保留率 93.95%）+ 4 时间特征 + 4 衍生特征 |
| **M2** 分析可视化 | 25 | 4 项分析共 7 张图，含自选项「拥堵指数（中位速度）」 |
| **M3** 预测模型 | 25 | PyTorch MLP vs 随机森林对比（结论：RF 在低维特征上 MAE 更低） |
| **M4** 问答接口 | 20 + 20 | 7 种规则问题类型 + DeepSeek LLM 兜底（System Prompt 经 3 轮迭代） |
| 人机协作报告 | 10 | 见 `人机协作报告.md`（约 2200 字，含 13 个真实 AI 犯错修复案例） |

---

## 输出产物（运行 main.py 后生成）

```
outputs/
├── m1_data_quality_report.md          # M1 质量报告（缺失率 + 9 类异常值）
├── m2_1_hourly_demand.png             # M2.1 分小时订单量
├── m2_1_weekday_demand.png            # M2.1 分星期订单量
├── m2_2_top10_zones.png               # M2.2 TOP 10 区域
├── m2_2_heatmap.png                   # M2.2 星期×小时热力图
├── m2_3_distance_fare_scatter.png     # M2.3 距离-车费散点
├── m2_3_fare_by_hour.png              # M2.3 各时段车费
├── m2_4_congestion_speed.png          # M2.4 拥堵指数
├── m3_loss_curve.png                  # M3 训练 loss 曲线
├── m3_pred_vs_truth.png               # M3 NN vs RF 预测对比
├── m3_model_comparison.md             # M3 模型对比报告
├── m4_system_prompt_design.md         # M4 LLM Prompt 三轮迭代过程
└── main_run_log.txt                   # main.py 完整运行日志
```

---

## M3 实验结果（核心发现）

| 模型 | MAE | RMSE |
|---|---:|---:|
| 神经网络 (PyTorch MLP, 隐层 128→64→32) | 26.71 | 49.09 |
| 随机森林 (n_estimators=100, max_depth=20) | **12.12** | **20.38** |

**随机森林显著优于神经网络**——在 6 维结构化特征 + 约 3.6 万样本的小数据规模下，深度学习容量过剩反而过拟合。这个反直觉结论是本次实验的重要发现，详见 `outputs/m3_model_comparison.md`。

---

## 技术栈

- **数据处理**：pandas + pyarrow
- **可视化**：matplotlib + seaborn
- **机器学习**：PyTorch（MLP）+ scikit-learn（RandomForest）
- **LLM 接口**：openai SDK（兼容 DeepSeek / Qwen / GLM 等）
- **运行环境**：Python 3.12 + Windows 10/11

---

## 已知问题与设计决策

1. **数据字段不含经纬度**：作业描述提到"上下客经纬度"，但 NYC TLC 自 2016 年起改用区域 ID（`PULocationID` / `DOLocationID`）。本项目通过 `taxi_zone_lookup.csv` 做了 ID→区域名映射。
2. **NN 表现不如 RF**：在 6 维结构化特征 + 3.6 万样本的小数据集上，深度学习容量过剩反而过拟合；如果引入天气、节假日等高维特征，NN 优势会更明显。
3. **GPU 未启用**：本项目刻意使用 CPU 版 PyTorch，便于无显卡环境复现。
4. **原始 parquet 数据未入库**：47 MB 超过 GitHub 网页 25 MB 单文件上限，请用 `download_data.py` 自动拉取。

---

## 提交说明

完整的人机协作过程记录、AI 犯错案例、三阶段（Native / Prompt / Vibe）对比、反思 —— 全部见 `人机协作报告.md`。
