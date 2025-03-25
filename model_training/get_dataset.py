import datetime

import polars as pl
import numpy as np


def get_future_price(ts, values, window=15):
    price_pointer = 0
    res = np.zeros(ts.size)
    for X_pointer in range(len(ts)):
        while price_pointer + 1 < len(ts) and ts[price_pointer] < ts[X_pointer] + window:
            price_pointer += 1
        res[X_pointer] = values[price_pointer]
    return res


def unnest(lst):
    return [item for sublist in lst for item in sublist]  


def compute_window_features(dataset):
    variables = ['o', 'c', 'v', 'vw']
    def rolling_aggs(t):
        return unnest([
        [
            pl.col(c).mean().alias(f"{c}_mean_{t}"),
            pl.col(c).std().alias(f"{c}_std_{t}"),
            pl.col(c).first().alias(f"{c}_lag_{t}"),
        ] for c in variables
    ])
    ewm_aggs = unnest([
        [
            pl.col(c).ewm_mean(com=2, ignore_nulls=False).alias(f"{c}_ewm_mean"),
            pl.col(c).ewm_std(com=2, ignore_nulls=False).alias(f"{c}_ewm_std"),
        ] for c in variables
    ])
    dataset = (
        dataset
        .drop("n")
        .with_columns(
            dataset
            .rolling(index_column="dttm", period="1d")
            .agg(*rolling_aggs("1d")),
        )
        .with_columns(
            dataset
            .rolling(index_column="dttm", period="3d")
            .agg(*rolling_aggs("3d")),
        )
        .with_columns(
            dataset
            .rolling(index_column="dttm", period="5d")
            .agg(*rolling_aggs("5d")),
        )
        .with_columns(
            dataset
            .rolling(index_column="dttm", period="15d")
            .agg(*rolling_aggs("15d")),
        )
        .with_columns(
            *ewm_aggs,
            (pl.col("o") - pl.col("o_lag_1d")).fill_nan(0).alias("o_diff_1d"),
            (pl.col("o") - pl.col("o_lag_3d")).fill_nan(0).alias("o_diff_3d"),
            (pl.col("o") - pl.col("o_lag_5d")).fill_nan(0).alias("o_diff_5d"),
            (pl.col("o") - pl.col("o_lag_15d")).fill_nan(0).alias("o_diff_15d"),
        )
    )
    dataset = dataset.fill_nan(0).fill_null(0)
    return dataset


def get_dataset_with_features(ticker="AAPL"):
    dataset = pl.read_parquet("stocks_1d_ohlcv.parquet")
    dataset = dataset.filter(pl.col("ticker") == ticker)
    dataset = dataset.with_columns(
        future_price=(
            get_future_price(
                dataset["t"].to_numpy(),
                dataset["vw"].to_numpy(),
                datetime.timedelta(days=2) / datetime.timedelta(milliseconds=1)
            )
        )
    )
    test_start_date = datetime.datetime(year=2025, month=1, day=1)
    dataset = dataset.with_columns(
        pl.when(pl.col("dttm") >= test_start_date).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split")
    )
    train, validation = compute_window_features(dataset).partition_by("split", maintain_order=True)
    train = train.sort("dttm")
    validation = validation.sort("dttm")
    return train, validation


def split_dataset_to_train_and_validation(train, validation):
    return (
        train.drop("future_price", "dttm", "t", "split", "ticker").to_pandas(),
        train["future_price"].to_pandas(),
        validation.drop("future_price", "dttm", "t", "split", "ticker").to_pandas(),
        validation["future_price"].to_pandas(),
    )


def get_timewindow_dataset(window_size=15, ticker="AAPL"):
    dataset = pl.read_parquet("stocks_1d_ohlcv.parquet")
    dataset = dataset.filter(pl.col("ticker") == ticker)
    dataset = dataset.with_columns(
        future_price=(
            get_future_price(
                dataset["t"].to_numpy(),
                dataset["vw"].to_numpy(),
                datetime.timedelta(days=2) / datetime.timedelta(milliseconds=1)
            )
        )
    )
    test_start_date = datetime.datetime(year=2025, month=1, day=1)
    dataset = dataset.with_columns(
        pl.when(pl.col("dttm") >= test_start_date).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split")
    )
    train, validation = dataset.partition_by("split", maintain_order=True)
    train = train.sort("dttm")
    validation = validation.sort("dttm")
    train_X = np.lib.stride_tricks.sliding_window_view(
        train.drop("future_price", "dttm", "t", "split", "ticker").to_numpy(),
        window_size,
        axis=0
    )
    train_y = train["future_price"].to_numpy()[window_size - 1:]
    validation_X = np.lib.stride_tricks.sliding_window_view(
        validation.drop("future_price", "dttm", "t", "split", "ticker").to_numpy(),
        window_size,
        axis=0
    )
    validation_y = validation["future_price"].to_numpy()[window_size - 1:]
    return train_X, train_y, validation_X, validation_y