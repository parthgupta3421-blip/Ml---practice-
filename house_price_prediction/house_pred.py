import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv("House_Price_Prediction_Dataset.csv")
df = df.drop(columns= ["Id"])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}  →  {df.shape[0]} rows, {df.shape[1]} columns")
print()
print(df.head())
print()
print("Data Types:")
print(df.dtypes)
print()
print("Statistical Summary:")
print(df.describe())
print()
print("Missing Values:")
print(df.isnull().sum())

# unique categorial column values 

print ()
print("unique Locations:", df["Location"].unique())
print("unique Conditions:",df["Condition"].unique())
print("Garage options:",df["Garage"].unique()) 

fig,axes = plt.subplots(2,3,figsize=(16,10))
plt.suptitle("HOUSE PRICE PREDICTION---EDA",fontsize = 15,fontweight="bold")

#price disterbution 
axes[0,0].hist(df["Price"],bins=40,color="steelblue",edgecolor="white",alpha=0.85)
axes[0,0].set_title("price prediction")
axes[0,0].set_xlabel("price($)")
axes[0,0].set_ylabel("count")

#scatter plotting 
axes[0,1].scatter(df["Area"],df["Price"],alpha = 0.3,color = "tomato",s=15)
axes[0,1].set_title("area vs space")
axes[0,1].set_xlabel("area(sq.ft)")
axes[0,1].set_ylabel("price($)")

#correlation matrix 
corr = df[["Area","Bedrooms","Bathrooms","Floors","YearBuilt","Price"]].corr()
sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",ax=axes[0,2],linewidths=0.5)
axes[0,2].set_title("correlation heatmap")

#avg price by location 
df.groupby("Location")["Price"].mean().sort_values().plot(kind = "barh",ax=axes[1,0],color="mediumseagreen")
axes[1,0].set_title("Avg Price by Location")
axes[1,0].set_xlabel("Price($)")

#avg prices by condition 
df.groupby("Condition")["Price"].mean().reindex(["Poor","Fair","Good","Excellent"]).plot(kind = "bar",ax=axes[1,1],color = "mediumpurple")
axes[1,1].set_title("Avg Price by Condition")
axes[1,1].tick_params(axis="x",rotation=0)

#price by bedrooms 
df.boxplot(column = "Price",by = "Bedrooms", ax= axes[1,2])
axes[1,2].set_title("Price by Bedrooms")
axes[1,2].set_xlabel("Bedrooms")

plt.sca(axes[1,2])
plt.title("Price by Bedrooms")

plt.tight_layout()
plt.show()

# pre processing (one hot encoding for current dataset)
df_model= df.copy()

df_model = pd.get_dummies(df_model,columns=["Location","Condition","Garage"],drop_first=True)

print ("columns after encoding")
print(df_model.columns.tolist())
print()
print(df_model.head())

#preprocessing done 

#========================================================================#

#train test split 

X= df_model.drop(columns=["Price"])
Y= df_model["Price"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print (f"training samples  :   {X_train.shape[0]}")
print (f"test samples  :   {X_test.shape[0]}")
print (f"no. of features  :   {X_train.shape[1]}")

# section 4 train the model 

model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)


print()
print("model trained successfully")
print(f"intecept value:  ${model.intercept_:,.2f}")


#evaluate the model 

# mae : mean absolute error 
# rmse : root mean squared error 
# R_squared / coefficent of determination 

mae = mean_absolute_error(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)
rmse= np.sqrt(mse)
r2= r2_score(Y_test,Y_pred)

print()
print("="*40)
print("MODEL EVALUATION")
print("="*40)
print(f"MAE:  ${mae:,.0f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"R2_SCORE: {r2:,.4f}")