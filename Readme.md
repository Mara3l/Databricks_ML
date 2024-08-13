# GoodData Dataricks ML POC flow

The core of the POC is a Streamlit application, which predicts future stock prices based on historical data. The app uses data fetched from GoodData and leverages a machine learning model hosted on Databricks to generate predictions.

## Flow Overview

1. Upload you data to GoodData. You can use the [stocks.csv](./stocks.csv) mock data and [GoodData Trial](www.gooddata.com/trial).
1. Fetch the data from GoodData in Databricks. Then create and register a model in the Unity Catalog. For this you can use the included [notebook](./Databricks_ARIMA_MODEL.ipynb).
1. Create a serving endpoint in Databricks.
1. Use the Databricks endpoint as well as the GoodData connection in the [steamlit application](./streamlit_app.py).

## Setup

All the needed python requirements are in the [requirements.txt](./requirements.txt). I've also included a [Makefile](./Makefile), to make your life a little easier, so with `make dev` you can create virtual env hassle-free.

Then you need to setup the data in GoodData, to learn how, refer to the _How do I get data to GoodData?_ section.

In databricks, you will need to provide connection to GoodData.

Specifically, you will need to provide these four:

```
host = "YOUR HOST"
token = "GD_TOKEN"
workspace_id = "GD_WORKSPACE"
visualization_id = "GD_VISUALIZATION"
```

Here is the relevant documentation to:

- [Create API Token](https://www.gooddata.com/docs/cloud/getting-started/create-api-token/)
- Understand the [Workspace and Visualization ID](https://www.gooddata.com/docs/cloud/create-workspaces/objects-identification/)

## Data Creation

The data was created with this python code:

```python
trend_start_price = 150
trend_end_price = 450
linear_trend = np.linspace(trend_start_price, trend_end_price, len(dates))
noise = np.random.normal(0, 20, len(dates))
prices = linear_trend + noise
```

If you want to create a new data, I've included [generate_data.py](./generate_data.py). It generates data till 7.7.2024, but feel free to tweak it.

## How do I get data to GoodData?

If you haven't uploaded any .csv files to GoodData so far, I highly recommend the official documentation on how to [Upload CSV Files](https://www.gooddata.com/docs/cloud/connect-data/csv/).

