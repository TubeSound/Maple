from __future__ import annotations

import sys
from dateutil  import tz

import csv
import os
import glob
from typing import TypedDict

import numpy as np
from datetime import datetime, timedelta

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

DATA_ROOT_AXIORY = "c:/develop/MarketData/axiory"
DATA_ROOT_HISTDATA = "c:/develop/MarketData/histdata"
# Backward compatibility for strategy modules that use the Axiory default.
DATA_ROOT = DATA_ROOT_AXIORY


class CandleArrays(TypedDict):
    jst: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

def str2datetime_jst(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=JST)
    else:
        dt = dt.astimezone(JST)
    return dt

class DataLoader:
    def __init__(self, symbol: str, data_root: str = DATA_ROOT_AXIORY):
        self.symbol = symbol
        self.data_root = data_root

    def file_path(self, timeframe: str, year: int, month: int) -> str:
        filename = f"{self.symbol}_{timeframe}_{year:04d}-{month:02d}.csv"
        timeframe_path = os.path.join(
            self.data_root,
            self.symbol,
            timeframe.upper(),
            filename,
        )
        if os.path.exists(timeframe_path):
            return timeframe_path

        # HistData files use a flat symbol directory:
        #   <root>/<symbol>/<symbol>_M1_YYYY-MM.csv
        flat_path = os.path.join(self.data_root, self.symbol, filename)
        return flat_path if os.path.exists(flat_path) else timeframe_path

    def tick_file_path(self, year: int, month: int) -> str:
        return self.file_path("TICK", year, month)

    def tick_pickle_file_paths(self, year: int, month: int) -> list[str]:
        """Return the first- and second-half pickle paths for one month."""

        filenames = [
            f"{self.symbol}_TICK_{year:04d}-{month:02d}-{part}.pkl"
            for part in (1, 2)
        ]
        directories = [
            os.path.join(self.data_root, self.symbol, "TICK"),
            os.path.join(self.data_root, self.symbol),
        ]
        for directory in directories:
            paths = [os.path.join(directory, filename) for filename in filenames]
            if any(os.path.exists(path) for path in paths):
                return paths
        return [os.path.join(directories[0], filename) for filename in filenames]
    
    def load_candle(self, timeframe: str, year: int, month_begin: int, month_end: int) -> pd.DataFrame:        
        import pandas as pd

        dfs = []
        for month in range(month_begin, month_end + 1):
            file_path = self.file_path(timeframe, year, month)
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist. Skipping.")
                continue
            
            df = pd.read_csv(file_path)
            df["jst"] = df["jst"].apply(str2datetime_jst)
            df["open"] = pd.to_numeric(df["open"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["jst", "open", "high", "low", "close"]).sort_values("jst")
            if df.empty:
                print(f"File {file_path} is empty after processing. Skipping.")
                continue
            dfs.append(df)    
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def load_candle_months(self, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        import pandas as pd

        start_time = start_time.astimezone(JST) if start_time.tzinfo else start_time.replace(tzinfo=JST)
        end_time = end_time.astimezone(JST) if end_time.tzinfo else end_time.replace(tzinfo=JST)

        dfs = []
        month = datetime(start_time.year, start_time.month, 1, tzinfo=JST)
        end_month = datetime(end_time.year, end_time.month, 1, tzinfo=JST)
        while month <= end_month:
            df = self.load_candle(timeframe, month.year, month.month, month.month)
            if not df.empty:
                dfs.append(df)
            if month.month == 12:
                month = datetime(month.year + 1, 1, 1, tzinfo=JST)
            else:
                month = datetime(month.year, month.month + 1, 1, tzinfo=JST)

        return pd.concat(dfs, ignore_index=True).sort_values("jst").reset_index(drop=True) if dfs else pd.DataFrame()

    def load_data(self, timeframe: str, start_time: datetime, end_time: datetime, how: str = "mt5") -> pd.DataFrame:
        import pandas as pd

        timeframe = timeframe.upper()
        start_time = start_time.astimezone(JST) if start_time.tzinfo else start_time.replace(tzinfo=JST)
        end_time = end_time.astimezone(JST) if end_time.tzinfo else end_time.replace(tzinfo=JST)

        dfs = []
        month = datetime(start_time.year, start_time.month, 1, tzinfo=JST)
        end_month = datetime(end_time.year, end_time.month, 1, tzinfo=JST)
        while month <= end_month:
            df = self.load_candle(timeframe, month.year, month.month, month.month)
            if not df.empty:
                dfs.append(df)
            if month.month == 12:
                month = datetime(month.year + 1, 1, 1, tzinfo=JST)
            else:
                month = datetime(month.year, month.month + 1, 1, tzinfo=JST)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True).sort_values("jst").reset_index(drop=True)
        jst = pd.to_datetime(df["jst"])
        return df.loc[(jst >= start_time) & (jst < end_time)].reset_index(drop=True)

    def load_tick(self, year: int, month_begin: int, month_end: int) -> pd.DataFrame:
        import pandas as pd

        dfs = []
        for month in range(month_begin, month_end + 1):
            pickle_paths = self.tick_pickle_file_paths(year, month)
            existing_pickle_paths = [
                path for path in pickle_paths if os.path.exists(path)
            ]
            if existing_pickle_paths:
                file_paths = existing_pickle_paths
                for missing_path in set(pickle_paths).difference(file_paths):
                    print(f"File {missing_path} does not exist. Skipping.")
            else:
                csv_path = self.tick_file_path(year, month)
                if not os.path.exists(csv_path):
                    print(
                        f"Files {pickle_paths} and {csv_path} do not exist. "
                        "Skipping."
                    )
                    continue
                file_paths = [csv_path]

            for file_path in file_paths:
                if file_path.lower().endswith(".pkl"):
                    df = pd.read_pickle(file_path)
                else:
                    df = pd.read_csv(file_path)

                required_columns = {"jst", "bid", "ask"}
                missing_columns = required_columns.difference(df.columns)
                if missing_columns:
                    print(
                        f"File {file_path} is missing columns: "
                        f"{', '.join(sorted(missing_columns))}. Skipping."
                    )
                    continue

                df["jst"] = pd.to_datetime(df["jst"], errors="coerce")
                if df["jst"].dt.tz is None:
                    df["jst"] = df["jst"].dt.tz_localize(JST)
                else:
                    df["jst"] = df["jst"].dt.tz_convert(JST)
                for column in ["bid", "ask"]:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                df = df.dropna(subset=["jst", "bid", "ask"])
                if df.empty:
                    print(f"File {file_path} is empty after processing. Skipping.")
                    continue
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True).sort_values("jst").reset_index(drop=True)

    def tick_session_paths(self) -> list[str]:
        """Daily JST sessions may include the following calendar day's ticks."""
        pattern = os.path.join(self.data_root, self.symbol, "TICK",
                               f"{self.symbol}_TICK_????-??-??.csv")
        return sorted(glob.glob(pattern))

    @staticmethod
    def read_tick_csv(path) -> pd.DataFrame:
        import pandas as pd
        frame = pd.read_csv(path)
        missing = {"jst", "bid", "ask"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing tick columns {sorted(missing)}")
        # Naive CSV times are JST, never UTC. Explicit offsets are respected.
        frame["jst"] = pd.to_datetime(
            frame["jst"].apply(str2datetime_jst), utc=True
        ).dt.tz_convert(JST)
        for name in ("bid", "ask"):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        if frame[["jst", "bid", "ask"]].isna().any().any() or not np.isfinite(frame[["bid", "ask"]]).all().all():
            raise ValueError(f"{path}: invalid tick timestamp or price")
        return frame.sort_values("jst", kind="stable").reset_index(drop=True)

    def latest_tick_time(self):
        paths = self.tick_session_paths()
        for path in reversed(paths):
            frame = self.read_tick_csv(path)
            if not frame.empty:
                return frame.jst.iloc[-1]
        raise ValueError("no tick CSV data")

    @staticmethod
    def tick_to_candle(ticks, seconds=10):
        """Left-labelled Bid OHLC; tick-count volume, no fabricated gap bars."""
        import pandas as pd
        if type(seconds) is not int or seconds <= 0:
            raise ValueError("seconds must be a positive integer")
        columns = ["jst", "open", "high", "low", "close", "volume"]
        if ticks.empty:
            return pd.DataFrame(columns=columns)
        grouped = ticks.sort_values("jst", kind="stable").set_index("jst").bid.resample(
            f"{seconds}s", origin="start_day", label="left", closed="left")
        bars = grouped.ohlc()
        bars["volume"] = grouped.count()
        return bars.dropna(subset=["open"]).reset_index()

    def load_tick_candles(self, start_time, end_time, seconds=10):
        """Load confirmed Tick-derived bars in [start, end).

        Include the preceding session file for post-midnight coverage. Do not
        infer the final bar's completion from the wall clock or filename.
        """
        import pandas as pd
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        start = start.tz_localize(JST) if start.tzinfo is None else start.tz_convert(JST)
        end = end.tz_localize(JST) if end.tzinfo is None else end.tz_convert(JST)
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError("start_time must precede end_time")
        parts = []
        for path in self.tick_session_paths():
            session = pd.Timestamp(os.path.basename(path).rsplit("_", 1)[-1][:-4], tz=JST)
            if session < start.normalize() - pd.Timedelta(days=1) or session > end:
                continue
            frame = self.read_tick_csv(path)
            if not frame.empty:
                parts.append(frame)
        if not parts:
            raise ValueError("no tick CSV data in the requested period")
        ticks = pd.concat(parts, ignore_index=True).sort_values("jst", kind="stable")
        # Last observed tick only confirms bars whose full interval has elapsed.
        cutoff = min(end, ticks.jst.iloc[-1].floor(f"{seconds}s"))
        ticks = ticks.loc[(ticks.jst >= start.floor(f"{seconds}s")) & (ticks.jst < cutoff)]
        bars = self.tick_to_candle(ticks, seconds)
        return bars.loc[(bars.jst >= start) & (bars.jst + pd.Timedelta(seconds=seconds) <= cutoff)].reset_index(drop=True)

    @staticmethod
    def _empty_candle_arrays() -> CandleArrays:
        return {
            "jst": np.array([], dtype=object),
            "open": np.array([], dtype=np.float64),
            "high": np.array([], dtype=np.float64),
            "low": np.array([], dtype=np.float64),
            "close": np.array([], dtype=np.float64),
            "volume": np.array([], dtype=np.float64),
        }

    def load_candle_numpy(
        self,
        timeframe: str,
        year: int,
        month: int,
    ) -> CandleArrays:
        """Load one candle CSV directly into a dictionary of NumPy arrays."""

        file_path = self.file_path(timeframe.upper(), year, month)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            return self._empty_candle_arrays()

        columns: dict[str, list] = {
            "jst": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        with open(file_path, encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"jst", "open", "high", "low", "close"}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{file_path} is missing columns: {', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    columns["jst"].append(str2datetime_jst(row["jst"]))
                    for name in ("open", "high", "low", "close"):
                        columns[name].append(float(row[name]))
                    volume = row.get("volume", "")
                    columns["volume"].append(
                        float(volume) if volume not in (None, "") else np.nan
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{file_path}:{line_number}: invalid candle row"
                    ) from exc

        return {
            "jst": np.asarray(columns["jst"], dtype=object),
            "open": np.asarray(columns["open"], dtype=np.float64),
            "high": np.asarray(columns["high"], dtype=np.float64),
            "low": np.asarray(columns["low"], dtype=np.float64),
            "close": np.asarray(columns["close"], dtype=np.float64),
            "volume": np.asarray(columns["volume"], dtype=np.float64),
        }

    def load_data_numpy(
        self,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> CandleArrays:
        """Load ``[start_time, end_time)`` as aligned NumPy arrays."""

        timeframe = timeframe.upper()
        start = (
            start_time.astimezone(JST)
            if start_time.tzinfo
            else start_time.replace(tzinfo=JST)
        )
        end = (
            end_time.astimezone(JST)
            if end_time.tzinfo
            else end_time.replace(tzinfo=JST)
        )
        if end <= start:
            raise ValueError("end_time must be later than start_time")

        chunks: list[CandleArrays] = []
        current = datetime(start.year, start.month, 1, tzinfo=JST)
        while current < end:
            chunk = self.load_candle_numpy(
                timeframe,
                current.year,
                current.month,
            )
            if len(chunk["jst"]):
                chunks.append(chunk)
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=JST)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=JST)

        if not chunks:
            return self._empty_candle_arrays()

        combined: CandleArrays = {
            name: np.concatenate([chunk[name] for chunk in chunks])
            for name in ("jst", "open", "high", "low", "close", "volume")
        }
        epoch_seconds = np.fromiter(
            (value.timestamp() for value in combined["jst"]),
            dtype=np.float64,
            count=len(combined["jst"]),
        )
        order = np.argsort(epoch_seconds, kind="stable")
        for name in combined:
            combined[name] = combined[name][order]

        mask = np.fromiter(
            (start <= value < end for value in combined["jst"]),
            dtype=np.bool_,
            count=len(combined["jst"]),
        )
        return {name: values[mask] for name, values in combined.items()}


def test():
    symbol = "JP225"
    timeframe = "M1"
    year = 2026
    month_begin = 1
    month_end = 3
    loader = DataLoader(symbol)
    df_candle = loader.load_candle(timeframe, year, month_begin, month_end)
    print(df_candle)
    
if __name__ == "__main__":
    test()

