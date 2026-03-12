import numpy as np 
import pandas as pd 

# importing data csv 
df = pd.read_csv("dirty_cafe_sales.csv")
print(df.head())
 
# checking missing values 
print("Empty values collunm wise")
print(df.isnull().sum()) 

print(df.info())

# replace missing values
df.replace("unknown",np.nan,inplace=True)

#convert numeric values
df["quantity"]=pd.to_numeric(df["quantity"],errors="coerce")
df["price per unit"]=pd.to_numeric(df["price per unit"],errors="coerce")
df["total spent"]=pd.to_numeric(df["total spent"],errors="coerce")

#handle missing values
df["quantity"].fillna(df["quantity"].median(),inplace=True)
df["price per unit"].fillna(df["price per unit"].median(),inplace=True)
df["total spent"].fillna(df["total spent"].median(),inplace=True)

df["item"].fillna(df["item"].mode()[0],inplace=True)
df["payment method"].fillna(df["payment method"].mode()[0],inplace=True)
df["location"].fillna(df["location"].mode()[0],inplace=True)

df["transaction date"]= pd.to_datetime(df["transaction date"])

#remove duplicates
df.drop_duplicates(inplace=True)

# checking 
df["total spent"]=df["quantity"]*df["price per unit"]