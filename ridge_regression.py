#multi collinearity problem 
#ridge regression 
#implementation of ridge regreesion 
import pandas as pd 
from sklearn.datasets import load_diabetes

data = load_diabetes()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.DataFrame(data.target,columns=["target"])

#now we split the data into the training set and tests splits 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#creating a ridge class 
from sklearn.linear_model import Ridge 
regressor=Ridge(alpha=1.0)
regressor.fit(x_train,y_train)

#lets make predictions on the test data
y_pred=regressor.predict(x_test)

#evaluate performance
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Test R2: {r2:.3f}")

#visualize the results
import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test["target"], y=y_pred.flatten(), mode="markers", name="Predicted vs Actual"))
fig.add_trace(go.Scatter(x=[0, 350], y=[0, 350], mode="lines", name="Ideal"))
fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
fig.show()