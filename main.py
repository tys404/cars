import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
headers = ["symboling", "normalized-losses", 'make', "fuel-type", "aspiration", "num-of-doors",
           "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight",
           "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
           "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv(url, header=None)
df.columns = headers
#
# print(df.head(5))
# print(df.dtypes)
# print(df.describe())
# print(df.all())
# print(df.describe(include="all"))
# print(df.info)
#
# print(df["symboling"])
# print(df["engine-type"])
# print(df["fuel-system"])

# delete rows without price
df.dropna(subset=["price"], axis=0, inplace=True)

# change '?' to NaN
df["normalized-losses"].replace('?', '', inplace=True)
mean = df["normalized-losses"].mean()
df["normalized-losses"].replace(np.nan, mean)

print(mean)



# df.to_csv("new.csv")

