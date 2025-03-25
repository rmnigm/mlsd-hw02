import requests
import datetime
import polars as pl
import plotly.graph_objects as go
import numpy as np


def intraday_data(ticker, from_dt, to_dt, token=None):
    print(f'Loading {ticker} data for {from_dt} - {to_dt}...')
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_dt}/{to_dt}?adjusted=true&sort=asc&limit=50000&apiKey={token}'
    r = requests.get(url)
    data = r.json()
    if 'results' in data:
        result = data['results']
        return result
    else:
        print(data["status"])
        print(data)
        return {}


def build_recent_dataset(tickers, token=None):
    from_timepoint = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime("%Y-%m-%d")
    to_timepoint = datetime.datetime.now().strftime("%Y-%m-%d")
    data = []
    for ticker in tickers:
        data.append(
            pl.DataFrame(intraday_data(ticker, from_timepoint, to_timepoint, token))
            .with_columns(
                pl.lit(ticker).alias("ticker"),
                pl.from_epoch(pl.col('t').cast(pl.Int64), time_unit='ms').alias("dttm"),
            )
        )
    return pl.concat(data)


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
    dataset = dataset
    return dataset


def get_dataset_with_features(ticker="GOOG", token=None):
    dataset = build_recent_dataset([ticker], token)
    dataset = compute_window_features(dataset)
    return dataset


def plot_predictions(predictions, dataset, timedelta, future_prices, days_to_plot=10):
    NORM_PPF_95 = 1.96
    ci_size = NORM_PPF_95 * np.std((future_prices - predictions)[:-timedelta.days])
    timepoints = dataset['dttm']
    shifted_timepoints = timepoints + timedelta
    
    shifted_timepoints = shifted_timepoints[-timedelta.days - 2:]
    predictions = predictions[-timedelta.days - 2:]
    timepoints = timepoints[-days_to_plot:]
    gt_values = dataset['vw'][-days_to_plot:]
    ticker = dataset['ticker'].unique()[0]
    fig = go.Figure(layout={'title': f'{ticker} volume-weighted price prediction'})
    fig.add_traces([
        go.Scatter(
            x=shifted_timepoints,
            y=predictions,
            mode='lines+markers',
            name='Predicted volume-weighted price'
        ),
        go.Scatter(
            x=timepoints,
            y=gt_values,
            mode='lines+markers',
            name='Actual volume-weighted price'
        ),
        go.Scatter(
            x = shifted_timepoints,
            y = predictions + ci_size,
            mode = 'lines',
            line_color = 'rgba(0,0,0,0)',
            showlegend = False
        ),
        go.Scatter(
            x = shifted_timepoints,
            y = predictions - ci_size,
            mode = 'lines',
            line_color = 'rgba(0,0,0,0)',
            name = '95% confidence interval',
            fill='tonexty',
            fillcolor = 'rgba(0, 0, 255, 0.1)')
    ])
    fig.show()