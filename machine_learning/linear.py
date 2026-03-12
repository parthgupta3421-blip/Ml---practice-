import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# step 1 creating a coplex dataset 
np.random.seed()
x = np.linspace(0,10,100) #independent variable 
y = 3*x**2 + 2*x+np.random.normal(0,10,100)

#splitting the data into the training set/testing set 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# step 3 implementing linear regression 
model= LinearRegression()
x_train = x_train.reshape(-1,1) #reshaping x_train to the 2d array 
model.fit (x_train,y_train) #training the model 

#step 4 make predictions
x_test = x_test.reshape(-1,1)
y_pred = model.predict(x_test)

#step 5 visualize the model 
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_test.flatten(), y=y_test,mode='markers',name='Actual Data'))
fig.add_trace(go.Scatter(x=x_test.flatten(),y=y_pred,mode='lines',name='Linear Regression line ='))

fig.update_layout(title='Linear Regression',
                  xaxis_title='Independent variable (x)',
                  yaxis_title='Dependent variable (y)')
fig.show()