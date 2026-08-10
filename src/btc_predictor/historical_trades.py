from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests


BINANCE_COLUMNS = (
    "trade_id", "price", "qty", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker",
)


def archive_url(exchange: str, day) -> str:
    day = pd.Timestamp(day).strftime("%Y-%m-%d")
    if exchange == "binance":
        template = os.getenv(
            "BINANCE_HISTORICAL_TRADES_URL",
            "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{date}.zip",
        )
    elif exchange == "bybit":
        template = os.getenv(
            "BYBIT_HISTORICAL_TRADES_URL",
            "https://s3.ap-southeast-1.amazonaws.com/public.bybit.com/trading/BTCUSDT/BTCUSDT{date}.csv.gz",
        )
    else:
        raise ValueError("exchange must be binance or bybit")
    return template.format(date=day)


def _download(url: str, destination: Path, timeout=60, attempts=3) -> str:
    last_error=None
    for attempt in range(max(1,int(attempts))):
        digest=hashlib.sha256()
        try:
            with requests.get(url,stream=True,timeout=timeout,headers={"User-Agent":"btc-flow-calibration/1.0"}) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024*1024):
                        if not block:continue
                        digest.update(block);handle.write(block)
            return digest.hexdigest()
        except requests.RequestException as exc:
            last_error=exc
            if attempt+1<attempts:time.sleep(2**attempt)
    raise last_error


def _normalize_binance(path: Path, chunksize=250_000):
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one Binance CSV member, found {len(members)}")
        with archive.open(members[0]) as source:
            for chunk in pd.read_csv(source, header=None, names=BINANCE_COLUMNS, usecols=range(7), chunksize=chunksize, low_memory=False):
                numeric_time = pd.to_numeric(chunk.transact_time, errors="coerce")
                valid = numeric_time.notna()
                if not valid.any():
                    continue
                chunk = chunk.loc[valid]
                maker = chunk.is_buyer_maker.astype(str).str.lower().isin(("true", "1"))
                yield pd.DataFrame({
                    "time": pd.to_datetime(numeric_time.loc[valid], unit="ms", utc=True),
                    "price": pd.to_numeric(chunk.price, errors="coerce"),
                    "qty": pd.to_numeric(chunk.qty, errors="coerce"),
                    "side": np.where(maker, "sell", "buy"),
                    "exchange": "binance",
                    "trade_id": chunk.trade_id.astype(str),
                }).dropna(subset=["time", "price", "qty"])


def _bybit_column(columns, *candidates):
    mapping = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    return None


def _normalize_bybit(path: Path, chunksize=250_000):
    with gzip.open(path, "rb") as source:
        for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            time_column = _bybit_column(chunk.columns, "timestamp", "time")
            price_column = _bybit_column(chunk.columns, "price")
            qty_column = _bybit_column(chunk.columns, "size", "qty", "quantity")
            side_column = _bybit_column(chunk.columns, "side")
            id_column = _bybit_column(chunk.columns, "trdMatchID", "trade_id", "id")
            if None in (time_column, price_column, qty_column, side_column):
                raise ValueError(f"Unsupported Bybit columns: {list(chunk.columns)}")
            raw_time = pd.to_numeric(chunk[time_column], errors="coerce")
            median = raw_time.dropna().median()
            unit = "ms" if median > 1e11 else "s"
            normalized = pd.DataFrame({
                "time": pd.to_datetime(raw_time, unit=unit, utc=True),
                "price": pd.to_numeric(chunk[price_column], errors="coerce"),
                "qty": pd.to_numeric(chunk[qty_column], errors="coerce"),
                "side": chunk[side_column].astype(str).str.lower(),
                "exchange": "bybit",
                "trade_id": chunk[id_column].astype(str) if id_column else chunk.index.astype(str),
            })
            yield normalized.loc[normalized.side.isin(("buy", "sell"))].dropna(subset=["time", "price", "qty"])


def normalize_day(exchange: str, day, root: str | Path, force=False) -> dict:
    """Stream one official archive into a normalized daily Parquet partition."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Historical normalization requires the research dependency pyarrow") from exc
    day = pd.Timestamp(day).floor("D")
    day = day.tz_localize("UTC") if day.tzinfo is None else day.tz_convert("UTC")
    partition = Path(root) / exchange / f"date={day.strftime('%Y-%m-%d')}"
    output = partition / "trades.parquet"
    manifest_path = partition / "manifest.json"
    if output.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())
    partition.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if exchange == "binance" else ".csv.gz"
    fd, raw_name = tempfile.mkstemp(prefix=f"{exchange}-{day:%Y-%m-%d}-", suffix=suffix, dir=partition)
    os.close(fd)
    raw_path = Path(raw_name)
    writer = None
    rows = 0
    minimum = None
    maximum = None
    minutes = set()
    try:
        digest = _download(archive_url(exchange, day), raw_path)
        iterator = _normalize_binance(raw_path) if exchange == "binance" else _normalize_bybit(raw_path)
        temp_output = output.with_suffix(".parquet.tmp")
        for frame in iterator:
            frame = frame.loc[(frame.time >= day) & (frame.time < day + pd.Timedelta(days=1))]
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_output, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
            chunk_min, chunk_max = frame.time.min(), frame.time.max()
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
            minutes.update(frame.time.dt.floor("min").astype(str).unique().tolist())
        if writer is not None:
            writer.close(); writer = None
        if rows == 0:
            raise ValueError(f"No normalized {exchange} trades for {day:%Y-%m-%d}")
        os.replace(temp_output, output)
        start = day
        complete = bool(minimum <= start + pd.Timedelta(minutes=1) and maximum >= start + pd.Timedelta(hours=23, minutes=59) and len(minutes) >= 1433)
        manifest = {
            "exchange": exchange,
            "date": day.strftime("%Y-%m-%d"),
            "source_url": archive_url(exchange, day),
            "archive_sha256": digest,
            "rows": rows,
            "minimum_time": minimum.isoformat(),
            "maximum_time": maximum.isoformat(),
            "minute_coverage": round(len(minutes) / 1440.0, 6),
            "complete": complete,
            "parquet": str(output),
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
        return manifest
    finally:
        if writer is not None:
            writer.close()
        raw_path.unlink(missing_ok=True)
        output.with_suffix(".parquet.tmp").unlink(missing_ok=True)


def normalize_range(start, end, root: str | Path, force=False, progress=None) -> list[dict]:
    start = pd.Timestamp(start).floor("D")
    end = pd.Timestamp(end).floor("D")
    manifests = []
    days = list(pd.date_range(start, end, freq="D", inclusive="left"))
    tasks=[(exchange,day) for day in days for exchange in ("binance","bybit")]
    total=len(tasks);done=0
    process_workers=max(0,min(4,int(os.getenv("HISTORICAL_PROCESS_WORKERS","0"))))
    workers=process_workers or max(1,min(4,int(os.getenv("HISTORICAL_DOWNLOAD_WORKERS","2"))))
    executor=ProcessPoolExecutor if process_workers else ThreadPoolExecutor
    with executor(max_workers=workers) as pool:
        futures={pool.submit(normalize_day,exchange,day,root,force): (exchange,day) for exchange,day in tasks}
        for future in as_completed(futures):
            manifest=future.result();manifests.append(manifest);done+=1
            if progress:progress(done,total,manifest)
    return sorted(manifests,key=lambda item:(item["date"],item["exchange"]))


def common_complete_days(root: str | Path, start, end) -> list[str]:
    root = Path(root)
    days = []
    for day in pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D", inclusive="left"):
        label = day.strftime("%Y-%m-%d")
        manifests = []
        for exchange in ("binance", "bybit"):
            path = root / exchange / f"date={label}" / "manifest.json"
            try:
                manifests.append(json.loads(path.read_text()))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                manifests.append({})
        if all(item.get("complete") for item in manifests):
            days.append(label)
    return days
