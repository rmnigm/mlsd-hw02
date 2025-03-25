# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "catboost>=1.2.7",
#   "click>=8.1.8",
#   "numpy>=1.26.4",
#   "polars>=1.25.2",
#   "requests>=2.32.3",
#   "joblib>=1.4.2",
#   "scikit-learn>=1.6.1",
#   "plotly>=5.24.0",
# ]
# ///


import datetime

import click
import joblib
import polars as pl

from utils import get_dataset_with_features, plot_predictions, get_future_price


@click.command()
@click.option('--datapath', type=str, required=False, help='Path to the pre-collected data in parquet format')
@click.option('--token', type=str, required=False, help='Token to use for the Polygon API')
@click.argument('ticker', type=str, required=True)
def report(*args, **kwargs):
    ticker = kwargs.get('ticker')
    data_path = kwargs.get('datapath')
    token = kwargs.get('token')
    assert token is not None or data_path is not None, "Token or data path must be provided"
    if token is not None:
        dataset = get_dataset_with_features(ticker, token)
        dataset.write_parquet(f"ohlcv_{ticker}.parquet")
    elif data_path is not None:
        dataset = pl.read_parquet(data_path).filter(pl.col("ticker") == ticker)
    future_prices = get_future_price(
        dataset['t'].to_numpy(),
        dataset['vw'].to_numpy(),
        datetime.timedelta(days=2) / datetime.timedelta(milliseconds=1)
    )
    print("Loading model...")
    model = joblib.load(f"model_{ticker}.pkl")
    print("Predicting...")
    predictions = model.predict(
        dataset.drop("dttm", "t", "ticker").to_numpy()
    )
    print("Plotting...")
    print(dataset["t"].head(10))
    time_delta = datetime.timedelta(days=2)
    plot_predictions(predictions, dataset, time_delta, future_prices)


if __name__ == "__main__":
    report()
