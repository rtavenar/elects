import pandas as pd

datasummary = pd.read_csv("DataSummary.csv")

(datasummary.Train + datasummary.Test).median()

gp = datasummary.loc[datasummary.Name=="GunPoint"]

pass