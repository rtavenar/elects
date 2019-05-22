import pandas as pd

df = pd.read_csv("runtimes.csv")

df["optimization start"] = pd.to_datetime(df["optimization start"])
df["optimization end"] = pd.to_datetime(df["optimization end"])

df["time"] = (df["optimization end"] - df["optimization start"]).astype('timedelta64[s]')

df.to_csv("parsed_runtimes.csv")
