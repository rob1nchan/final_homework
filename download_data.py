"""
download_data.py
下载纽约市黄色出租车 2023 年 1 月的行程数据 + 区域查找表。
数据来源: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
"""
from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FILES = [
    {
        "url": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
        "filename": "yellow_tripdata_2023-01.parquet",
        "desc": "黄色出租车 2023年1月 行程数据 (~47MB，~300万条)",
    },
    {
        "url": "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        "filename": "taxi_zone_lookup.csv",
        "desc": "区域查找表（LocationID 映射到区域名）",
    },
]


def download(url: str, filepath: Path, desc: str) -> None:
    """带进度显示的流式下载，已存在则跳过。"""
    if filepath.exists():
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"✓ 已存在: {filepath.name} ({size_mb:.1f} MB) - 跳过")
        return

    print(f"↓ 正在下载: {desc}")
    print(f"  URL: {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    done = 0
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                print(f"\r  进度: {pct:5.1f}%  ({done/1024/1024:6.1f} / {total/1024/1024:6.1f} MB)",
                      end="", flush=True)
    print(f"\n✓ 完成: {filepath.name}\n")


def main():
    print(f"数据保存目录: {DATA_DIR}\n")
    for f in FILES:
        download(f["url"], DATA_DIR / f["filename"], f["desc"])
    print("全部数据下载完成 ✓")


if __name__ == "__main__":
    main()