import pandas as pd

'''data_utara = pd.read_csv('data-utara(06-09).csv')
data_selatan = pd.read_csv('data-selatan(06-09).csv')
data_timur = pd.read_csv('data-timur(06-09).csv')
data_barat = pd.read_csv('data-barat(06-09).csv')
data_gabungan = [data_utara, data_selatan, data_timur, data_barat]
result = pd.concat(data_gabungan)
result.to_csv('data_gab(06-09).csv', index=False)'''

data_vehicle = "data_gab(06-09).csv"
data = pd.read_csv(data_vehicle)
data1 = data.sort_values(by=["Time"])
data1.to_csv("data_gab(06-09)-sort.csv", index=False)