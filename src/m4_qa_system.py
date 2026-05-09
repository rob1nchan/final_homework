"""
src/m4_qa_system.py
M4 自然语言问答接口（命令行循环）
- 7 种问题类型规则匹配（作业要求 ≥ 5 种）
- 每个回复返回：数字结论 + 相关图表路径
- 留好 LLM 兜底接口（下一步实现 +20 加分）
"""

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ====== LLM 配置（选做加分） ======
# 优先从 secrets_config 读 key；读不到则禁用 LLM，只走规则匹配。
try:
    from secrets_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

    USE_LLM = bool(LLM_API_KEY) and not LLM_API_KEY.startswith("sk-在这里")
except ImportError:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL = "", "", ""
    USE_LLM = False

# System Prompt（最终版 V3，迭代过程见 outputs/m4_system_prompt_design.md）
SYSTEM_PROMPT = """你是"纽约出租车数据分析助手"，专门回答关于 2023 年 1 月纽约黄色出租车的问题。

【已有数据范围】
- 时间：2023年1月，约 290 万条清洗后的行程记录
- 字段：上下车区域、行程时长/距离、车费、乘客数、支付方式
- 已预计算：分小时/星期订单量、TOP 10 区域、分时段平均车费、分时段中位速度

【回答规则】
1. 问题属于上述范围 → 基于常识用中文简短作答，避免编造具体数字（建议用户用规则查询拿精确数）
2. 问题超出范围（其他城市/其他年份/天气/个人信息）→ 直接说明无法回答，给一句替代建议
3. 控制在 80 字以内，不要长篇免责声明

【禁区】
- 不报具体精确的统计数字
- 不预测真实未来需求（数据只到 2023年1月）
- 不讨论与本数据无关的话题
"""
WEEKDAY_KEYWORDS = {
    "周一": 0, "星期一": 0, "礼拜一": 0,
    "周二": 1, "星期二": 1, "礼拜二": 1,
    "周三": 2, "星期三": 2, "礼拜三": 2,
    "周四": 3, "星期四": 3, "礼拜四": 3,
    "周五": 4, "星期五": 4, "礼拜五": 4,
    "周六": 5, "星期六": 5, "礼拜六": 5,
    "周日": 6, "星期日": 6, "礼拜日": 6, "周天": 6,
}

# 区域中文别名 → (LocationID, 英文名)
ZONE_ALIASES = {
    "JFK": (132, "JFK Airport"),
    "拉瓜迪亚": (138, "LaGuardia Airport"),
    "LGA": (138, "LaGuardia Airport"),
    "纽瓦克": (1, "Newark Airport"),
    "EWR": (1, "Newark Airport"),
    "时代广场": (230, "Times Sq/Theatre District"),
    "中央公园": (43, "Central Park"),
    "曼哈顿下城": (87, "Financial District North"),
}


class QASystem:
    """规则匹配为主的问答系统。"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.zone_map = self._load_zone_map()
        self._precompute()
        # 7 种问题类型 → 关键词 → 处理函数
        self.handlers = [
            ("时段/高峰查询", ["几点", "什么时候", "高峰", "最忙", "订单最多", "最少"], self.handle_time_demand),
            ("区域排名", ["哪个区", "哪些区", "top", "TOP", "前几", "上车最多", "下车最多", "最热门"],
             self.handle_zone_ranking),
            ("需求预测", ["预测", "估计", "可能有多少", "大概多少订单", "大概有多少"], self.handle_demand_predict),
            ("车费估算", ["多少钱", "费用", "车费", "几块"], self.handle_fare_estimate),
            ("拥堵/速度", ["最堵", "拥堵", "速度", "几点最慢"], self.handle_congestion),
            ("工作日 vs 周末", ["工作日", "周末", "周中", "对比"], self.handle_weekday_compare),
            ("数据概览", ["总共", "一共", "多少条", "数据规模", "数据量"], self.handle_overview),
        ]
        self.llm_client = self._init_llm()  # LLM 兜底（启用条件：secrets_config 配好 key）

    def _init_llm(self):
        if not USE_LLM:
            return None
        try:
            from openai import OpenAI
            return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        except Exception as e:
            print(f"[LLM 初始化失败] {e}")
            return None

    def _load_zone_map(self):
        zones = pd.read_csv(DATA_DIR / "taxi_zone_lookup.csv")
        return dict(zip(zones["LocationID"], zones["Zone"]))

    def _precompute(self):
        """启动时一次性算好所有常用聚合，加快每次问答响应速度。"""
        df = self.df
        day_counts = df.groupby("is_weekend")["date"].nunique()
        h = df.groupby(["is_weekend", "hour"]).size().reset_index(name="trips")
        h["avg"] = h.apply(lambda r: r["trips"] / day_counts.get(r["is_weekend"], 1), axis=1)
        self.hourly_demand = h
        self.top_pu = df["PULocationID"].value_counts().head(10)
        self.top_do = df["DOLocationID"].value_counts().head(10)
        self.fare_per_mile_median = df["fare_per_mile"].median()
        valid = df[df["trip_speed_mph"].between(1, 60)]
        self.speed_by_hour = valid.groupby(["is_weekend", "hour"])["trip_speed_mph"].median()
        daily = df.groupby(["day_of_week", "date"]).size().reset_index(name="trips")
        self.weekday_avg = daily.groupby("day_of_week")["trips"].mean()
        self.total = len(df)

    # ====== 7 个处理器 ======
    def handle_time_demand(self, q):
        is_weekend = 1 if "周末" in q else 0
        sub = self.hourly_demand[self.hourly_demand["is_weekend"] == is_weekend]
        type_name = "周末" if is_weekend else "工作日"
        peak = sub.loc[sub["avg"].idxmax()]
        valley = sub.loc[sub["avg"].idxmin()]
        return {
            "answer": f"【时段查询】{type_name}订单最多的时段是 {int(peak['hour']):02d}:00 "
                      f"(平均 {peak['avg']:.0f} 单/小时)，最少是 {int(valley['hour']):02d}:00 "
                      f"(平均 {valley['avg']:.0f} 单/小时)。",
            "chart": "outputs/m2_1_hourly_demand.png",
        }

    def handle_zone_ranking(self, q):
        m = re.search(r"(\d+)", q)
        n = min(int(m.group(1)) if m else 5, 10)
        is_drop = any(k in q for k in ["下车", "下客", "目的地", "终点"])
        series = self.top_do if is_drop else self.top_pu
        verb = "下车" if is_drop else "上车"
        lines = [
            f"  {i + 1}. {self.zone_map.get(zid, f'Zone {zid}')} ({cnt:,} 次)"
            for i, (zid, cnt) in enumerate(series.head(n).items())
        ]
        return {
            "answer": f"【区域排名】{verb}最多的 TOP {n} 区域：\n" + "\n".join(lines),
            "chart": "outputs/m2_2_top10_zones.png",
        }

    def handle_demand_predict(self, q):
        dow = next((v for k, v in WEEKDAY_KEYWORDS.items() if k in q), None)
        hm = re.search(r"(\d{1,2})\s*点", q)
        hour = int(hm.group(1)) if hm else None
        if hour is not None and "下午" in q and hour < 12:
            hour += 12
        zone_id = next((zid for alias, (zid, _) in ZONE_ALIASES.items() if alias in q), None)

        if dow is None and hour is None and zone_id is None:
            return {"answer": "请提供更具体的条件，如 '周一早上 8 点 JFK 大概多少订单'。", "chart": None}

        df = self.df
        cond = pd.Series(True, index=df.index)
        desc = []
        if hour is not None:
            cond &= (df["hour"] == hour);
            desc.append(f"{hour:02d}:00")
        if dow is not None:
            cond &= (df["day_of_week"] == dow);
            desc.append(["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dow])
        if zone_id is not None:
            cond &= (df["PULocationID"] == zone_id);
            desc.append(self.zone_map.get(zone_id, f"Zone {zone_id}"))

        sub = df[cond]
        if len(sub) == 0:
            return {"answer": "该条件下历史无数据，无法预测。", "chart": None}
        n_groups = max(sub["date"].nunique(), 1)
        avg = len(sub) / n_groups
        return {
            "answer": f"【需求预测】基于历史均值，{' × '.join(desc)} 的预计订单量约为 {avg:.0f} 单。",
            "chart": "outputs/m3_pred_vs_truth.png",
        }

    def handle_fare_estimate(self, q):
        dm = re.search(r"(\d+(?:\.\d+)?)\s*(英里|公里|km|mile)", q, re.IGNORECASE)
        if dm:
            dist = float(dm.group(1))
            if dm.group(2).lower() in ("公里", "km"):
                dist /= 1.609
            est = dist * self.fare_per_mile_median
            return {
                "answer": f"【车费估算】历史中位单价 ${self.fare_per_mile_median:.2f}/英里，"
                          f"{dist:.1f} 英里行程预计约 ${est:.2f}（不含小费/过路费/拥堵附加）。",
                "chart": "outputs/m2_3_distance_fare_scatter.png",
            }
        zones_found = [(zid, n) for alias, (zid, n) in ZONE_ALIASES.items() if alias in q]
        if len(zones_found) >= 2:
            (z1, n1), (z2, n2) = zones_found[0], zones_found[1]
            sub = self.df[(self.df["PULocationID"] == z1) & (self.df["DOLocationID"] == z2)]
            if len(sub) == 0:
                sub = self.df[(self.df["PULocationID"] == z2) & (self.df["DOLocationID"] == z1)]
            if len(sub) > 0:
                return {
                    "answer": f"【车费估算】{n1} ↔ {n2} 历史平均车费 ${sub['fare_amount'].mean():.2f}"
                              f" (基于 {len(sub):,} 条记录)。",
                    "chart": "outputs/m2_3_distance_fare_scatter.png",
                }
        return {
            "answer": "请提供距离 (例: '5 英里多少钱') 或两个明确区域 (例: 'JFK 到时代广场多少钱')。",
            "chart": None,
        }

    def handle_congestion(self, q):
        is_weekend = 1 if "周末" in q else 0
        type_name = "周末" if is_weekend else "工作日"
        speed = self.speed_by_hour.xs(is_weekend)
        slowest, fastest = speed.idxmin(), speed.idxmax()
        return {
            "answer": f"【拥堵分析】{type_name}最堵时段为 {slowest:02d}:00 (中位速度 "
                      f"{speed[slowest]:.1f} mph)，最畅通为 {fastest:02d}:00 ({speed[fastest]:.1f} mph)。",
            "chart": "outputs/m2_4_congestion_speed.png",
        }

    def handle_weekday_compare(self, q):
        wd = self.weekday_avg.iloc[:5].mean()
        we = self.weekday_avg.iloc[5:].mean()
        winner = "工作日" if wd > we else "周末"
        ratio = max(wd, we) / min(wd, we)
        return {
            "answer": f"【时段对比】2023年1月，工作日日均 {wd:,.0f} 单，周末日均 {we:,.0f} 单。"
                      f"{winner}订单更多，约为对方的 {ratio:.2f} 倍。",
            "chart": "outputs/m2_1_weekday_demand.png",
        }

    def handle_overview(self, q):
        return {
            "answer": f"【数据概览】2023年1月纽约黄色出租车清洗后共 {self.total:,} 条订单，"
                      f"覆盖 {self.df['PULocationID'].nunique()} 个上车区域。",
            "chart": "outputs/m1_data_quality_report.md",
        }

    # ====== 主入口 ======
    def answer(self, question: str) -> dict:
        # 解释类问题（"为什么/为啥/原因/解释"）优先走 LLM，
        # 避免被规则关键词误匹配（例如"为什么晚高峰订单多"含"高峰"会被时段查询截胡）
        explanation_keywords = ["为什么", "为啥", "为何", "原因", "怎么解释", "解释一下", "如何理解"]
        if any(k in question for k in explanation_keywords) and self.llm_client:
            return self._llm_fallback(question)

        # 规则匹配（M4 基础）
        for name, kws, handler in self.handlers:
            if any(k in question for k in kws):
                try:
                    res = handler(question)
                    res["matched"] = name
                    return res
                except Exception as e:
                    return {"answer": f"处理 [{name}] 时出错：{e}", "chart": None, "matched": name}

        # 都不匹配 → LLM 兜底
        if self.llm_client:
            return self._llm_fallback(question)
        return {
            "answer": "抱歉，无法识别问题类型。可以尝试问：\n"
                      "  · 几点订单最多？\n  · TOP 5 上车区域？\n  · 5 英里大概多少钱？\n"
                      "  · 几点最堵？\n  · 工作日 vs 周末哪个订单多？",
            "chart": None,
            "matched": "未匹配",
        }
        for name, kws, handler in self.handlers:
            if any(k in question for k in kws):
                try:
                    res = handler(question)
                    res["matched"] = name
                    return res
                except Exception as e:
                    return {"answer": f"处理 [{name}] 时出错：{e}", "chart": None, "matched": name}
        # 都不匹配 → LLM 兜底（下一步实现）
        if self.llm_client:
            return self._llm_fallback(question)
        return {
            "answer": "抱歉，无法识别问题类型。可以尝试问：\n"
                      "  · 几点订单最多？\n  · TOP 5 上车区域？\n  · 5 英里大概多少钱？\n"
                      "  · 几点最堵？\n  · 工作日 vs 周末哪个订单多？",
            "chart": None,
            "matched": "未匹配",
        }

    def _llm_fallback(self, question: str) -> dict:
        """规则没匹配上时，转给 LLM 给出解释性回复。"""
        try:
            resp = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return {
                "answer": "【LLM 解释】" + resp.choices[0].message.content.strip(),
                "chart": None,
                "matched": f"LLM ({LLM_MODEL})",
            }
        except Exception as e:
            return {"answer": f"LLM 调用失败: {e}", "chart": None, "matched": "LLM 错误"}


# ====== 命令行循环 ======
def chat_loop(qa: QASystem):
    print("\n" + "=" * 60)
    print("纽约出租车数据问答系统  (输入 'q' 或 '退出' 结束)")
    print("=" * 60)
    print(f"LLM 兜底: {'已启用' if qa.llm_client else '未启用 (基础版)'}")
    print("\n示例问题:")
    print("  - 几点订单最多？工作日呢？周末呢？")
    print("  - TOP 10 上车区域？")
    print("  - 周五晚上 7 点时代广场大概多少订单？")
    print("  - 8 英里大概多少钱？")
    print("  - 几点最堵？")
    print("  - 工作日和周末哪个订单多？")
    print("  - 总共有多少条数据？")
    print()

    while True:
        try:
            q = input(">> 你想问什么？  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!");
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit", "退出"):
            print("再见!");
            break
        res = qa.answer(q)
        print(f"\n[匹配类型: {res.get('matched', '未知')}]")
        print(res["answer"])
        if res.get("chart"):
            print(f"[相关图表] {res['chart']}")
        print()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from m1_data_processing import process_pipeline

    df = process_pipeline(use_cache=True)
    qa = QASystem(df)
    chat_loop(qa)
