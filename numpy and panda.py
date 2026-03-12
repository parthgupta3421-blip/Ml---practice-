import numpy as np 
import pandas as pd 

# importing data csv 
df = pd.read_csv("dirty_cafe_sales.csv")

# standardize column names to lowercase without spaces for easier access
# e.g. 'Price Per Unit' -> 'price_per_unit'
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(' ', '_')
)

print(df.head())
 
# checking missing values 
print("Empty values collunm wise")
print(df.isnull().sum()) 

print(df.info())

# replace missing values
df.replace("unknown",np.nan,inplace=True)

#convert numeric values
# use normalized column names
numeric_cols = ["quantity", "price_per_unit", "total_spent"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

#handle missing values
# numeric columns
for col in ["quantity", "price_per_unit", "total_spent"]:
    df[col] = df[col].fillna(df[col].median())

# categorical columns
for col in ["item", "payment_method", "location"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# convert dates with coercion for invalid entries
# using assignment on DataFrame directly avoids chained assignment warnings
df = df.assign(transaction_date=pd.to_datetime(df["transaction_date"], errors="coerce"))

# fill any remaining NaT values with the median date so all rows have a timestamp
df["transaction_date"] = df["transaction_date"].fillna(df["transaction_date"].median())

#remove duplicates
df.drop_duplicates(inplace=True)

# checking 
# recalc total spent after cleaning (columns are normalized)
df["total_spent"] = df["quantity"] * df["price_per_unit"]

#final check 
print(df.info())
print(df.describe())
print(df.isnull().sum())

#save clean dataset 
df.to_csv("clean_transactions.csv",index=False)