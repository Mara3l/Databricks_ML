import os
import requests
import numpy as np
import pandas as pd
import json
from gooddata_pandas import GoodPandas
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("GOODDATA_HOST")
token = os.getenv("GOODDATA_TOKEN")
workspace_id = os.getenv("GOODDATA_WORKSPACE_ID")
visualization_id = os.getenv("VISUALIZATION_ID")
databricks_url = os.getenv("DATABRICKS_URL")
databricks_token = os.getenv("DATABRICKS_TOKEN")


@st.cache_data
def get_dataframe(visualization_id: str):
    gp = GoodPandas(host, token)

    frames = gp.data_frames(workspace_id)
    return frames.for_visualization(visualization_id)


# Function for Databricks parse
def create_tf_serving_json(data):
    return {
        "inputs": (
            {name: data[name].tolist() for name in data.keys()}
            if isinstance(data, dict)
            else data.tolist()
        )
    }


@st.cache_data
def score_model(dataset):
    url = databricks_url
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    dataset.index = dataset.index.astype(str)

    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )
    return response.json()


def init_app():
    df = get_dataframe(visualization_id)

    st.title("Net Sales Prediction")

    to_predict = st.slider(
        "Number of months to predict:", min_value=1, max_value=24, value=10
    )

    last_date = df.index[-1]
    new_dates = pd.date_range(last_date, periods=to_predict + 1, freq="MS")[1:]

    zeroes = [0 for _ in range(to_predict)]
    pred_df = pd.DataFrame(zeroes, index=new_dates, columns=["Stock_Price"])

    result = score_model(pred_df)
    if "predicted_mean" in result["predictions"][0]:
        predictions = [item["predicted_mean"] for item in result["predictions"]]
    else:
        predictions = [
            value for item in result["predictions"] for value in item.values()
        ]

    pred_df = pd.DataFrame(predictions, index=new_dates, columns=["Stock_Price"])

    new_df = pd.concat([df, pred_df])

    st.write("### Stock Price Over Time with Predictions")

    fig = px.line(
        new_df,
        x=new_df.index,
        y="Stock_Price",
        title="Net Sales Over Time with Predictions",
    )

    gap_df = pd.concat([df.tail(1), pred_df.head(1)])

    fig.add_scatter(
        x=pred_df.index,
        y=pred_df["Stock_Price"],
        mode="lines",
        name="Predictions",
        line=dict(color="red"),
    )
    fig.add_scatter(
        x=gap_df.index,
        y=gap_df["Stock_Price"],
        mode="lines",
        name="Predictions",
        line=dict(color="red"),
    )
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Historical Data")
        st.dataframe(df)

    with col2:
        st.write("### Predictions")
        st.dataframe(pred_df)


init_app()
