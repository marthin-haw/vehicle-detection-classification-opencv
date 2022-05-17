import pandas as pd

data_vehicle = "data1.csv"
data = pd.read_csv(data_vehicle)
data1 = data.sort_values(by=["Time"])
data1.to_csv("data1-sort.csv", index=False)