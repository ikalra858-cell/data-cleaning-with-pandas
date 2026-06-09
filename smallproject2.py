import pandas as pd
import numpy as np

data = {
"Customer": [" aman", "Riya", "raj", "Aman", "riya ", "Karan", "Raj"],
"Age": [23, 25, None, 23, 25, 30, None],
"Product": ["Laptop", "Phone", "Tablet", "Laptop", "Phone", "Laptop", "Tablet"],
"Price": [70000, 20000, 15000, 70000, 20000, None, 15000],
"Quantity": [1, 2, 1, 1, 2, 1, None]
}

df = pd.DataFrame(data)
print(df)

# data exploration
print("Exploring Data")
print(df.head())
df.info()
print(df.isnull().sum())

# data cleaning 
print("Cleaning Data")
df["Customer"] = df["Customer"].str.strip()
df["Customer"] = df["Customer"].str.capitalize()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Price"] = df["Price"].fillna(df["Price"].mean())
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].mean())
print(df)

# fix data issues 
print("fixing data issues")
df = df.drop_duplicates(subset=["Customer","Product"],keep="first")
print(df)

# feature engineering
print("Feature Engineering")
df["Total_Sale"] = df["Price"] * df["Quantity"]
print(df)

# Data Analysis
print("Analyzing data")
print(df.loc[df["Total_Sale"].idxmax(),"Customer"])
print(df[df["Total_Sale"]>40000])
df.sort_values(by="Total_Sale",ascending=False,inplace=True)
print(df)
print(df[["Customer","Product","Total_Sale"]])

# numpy practice
print("Converting into arrays")
array = df[["Price","Quantity"]].to_numpy()
print(array)
print(array.mean())
print(array.min())
print(array.max())