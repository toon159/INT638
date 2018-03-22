import pandas as pd

df = pd.read_csv('./titanic_data_set/train.csv')
print(df.head())
print(df.shape)
print('columns' + df.columns)
print(df.describe())
print(df.info())