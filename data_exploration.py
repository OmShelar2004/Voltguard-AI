import pandas as pd 
import matplotlib.pyplot as plt 


df = pd.read_csv("ev_battery_dataset.csv")
print(df.head())

#exploring the dataset

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())


#visualizing the dataset using histogram

df.hist(figsize=(10,8))

plt.tight_layout()

plt.show()

#identifying the correlation

print(df.corr())