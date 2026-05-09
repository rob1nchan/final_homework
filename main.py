"""
main.py
纽约出租车数据分析与智能问答系统 — 一键运行入口
执行顺序：M1 数据处理 → M2 可视化 → M3 模型训练 → M4 问答交互
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from m1_data_processing import process_pipeline
from m2_visualization import run_all_visualizations
from m3_prediction import run_m3
from m4_qa_system import QASystem, chat_loop


def main():
    print("\n" + "=" * 70)
    print("  纽约出租车出行数据分析与智能问答系统")
    print("  《人工智能编程语言》期末大作业  |  周婧扬 / 25361053")
    print("=" * 70)

    t0 = time.time()
    print("\n>>> [M1] 数据处理（加载、质量报告、清洗、特征工程）")
    df = process_pipeline(use_cache=True)
    print(f"    M1 完成，耗时 {time.time() - t0:.1f} 秒")

    t1 = time.time()
    print("\n>>> [M2] 分析可视化（4 项分析，共 7 张图）")
    run_all_visualizations(df)
    print(f"    M2 完成，耗时 {time.time() - t1:.1f} 秒")

    t2 = time.time()
    print("\n>>> [M3] 神经网络预测 + 随机森林对比")
    run_m3(df)
    print(f"    M3 完成，耗时 {time.time() - t2:.1f} 秒")

    print(f"\n>>> [M4] 智能问答交互（总预热 {time.time() - t0:.1f} 秒）")
    qa = QASystem(df)
    chat_loop(qa)


if __name__ == "__main__":
    main()