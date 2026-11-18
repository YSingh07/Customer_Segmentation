import pandas as pd
import numpy as np

data =  pd.read_csv("data/Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.duplicated().sum())
print(data.isnull().sum())
print(data.describe())