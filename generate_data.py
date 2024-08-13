import numpy as np
import pandas as pd

data = {
    "Date": [
        "2022-02-18",
        "2022-03-20",
        "2022-04-19",
        "2022-05-19",
        "2022-06-18",
        "2022-07-18",
        "2022-08-17",
        "2022-09-16",
        "2022-10-16",
        "2022-11-15",
        "2022-12-15",
        "2023-01-14",
        "2023-02-13",
        "2023-03-15",
        "2023-04-14",
        "2023-05-14",
        "2023-06-13",
        "2023-07-13",
        "2023-08-12",
        "2023-09-11",
        "2023-10-11",
        "2023-11-10",
        "2023-12-10",
        "2024-01-09",
        "2024-02-08",
        "2024-03-09",
        "2024-04-08",
        "2024-05-08",
        "2024-06-07",
        "2024-07-07",
    ]
}
trend_start_price = 150
trend_end_price = 450
linear_trend = np.linspace(trend_start_price, trend_end_price, len(data["Date"]))
noise = np.random.normal(0, 20, len(data["Date"]))
prices = linear_trend + noise

mock_stock_data = pd.DataFrame({"Date": data["Date"], "Stock Price": prices})

trend_stock_file_path = "mock_stock.csv"
mock_stock_data.to_csv(trend_stock_file_path, index=False)
