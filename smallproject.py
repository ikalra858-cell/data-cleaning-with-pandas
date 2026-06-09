import pandas as pd
import numpy as np

data = {
    "Name": ["Aman", "Sara", "John", "Riya", "Arjun", "Sara ", "john", "Kiran", None, "Aman"],
    "Age": [21, 22, None, 23, 21, 22, 20, None, 24, 21],
    "Math": [88, 92, 79, None, 85, 92, 79, 95, 67, 88],
    "Science": [90, 85, None, 91, 87, 85, 88, 93, None, 90],
    "English": [85, None, 78, 89, 84, 82, 78, 91, 65, None]
}

df = pd.DataFrame(data)
print(df)

# Part-1 Data Exploration
print("------Data Exploration------")
print(df.head())
df.info()
print(df.isnull().sum())

# Part-2 Cleaning the Data
print("------Cleaning Data------")
df["Name"] = df["Name"].str.strip()
df["Name"] = df["Name"].str.capitalize()
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Math"] = df["Math"].fillna(df["Math"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())
print(df)

# Part-3 Fix Data Issues 
print("------Dropping Duplicates------")
df = df.drop_duplicates(subset = ["Name"])
print(df)

# Part-4 Feature Engineering 
print("------Feature Engineering------")
df["Total"] = df["Math"] + df["Science"] + df["English"]
df["Average"] = df["Total"]/3
print(df)

# Part-5 Data Analysis 
print("------Data Analyzing------")
print(df.loc[df["Average"].idxmax(), "Name"])
print(df[df["Average"]>85])
df.sort_values(by="Total",ascending=False,inplace=True)
print(df)
print(df[["Name","Total"]])

# Part-6 NumPy practice 
print("------Converting marks into NumPy array------")
marks = df[["Math","Science","English"]].values
print(marks)
print(marks.mean())
print(marks.min())
print(marks.max())