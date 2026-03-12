import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#numpy seed
np.random.seed(0)
x = np.linspace(0,10,100)#independent variable 
y = 3*x**2+2*x+np.random.normal(0,10,100)#dependent variable 

#reshape x and y 
x = x.reshape(-1,1)
y = y.reshape(-1,1)

#split data into the training set
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#transform independent variable x into polynomial features 
poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)

#model creation 
model=LinearRegression()
model.fit(x_train_poly,y_train)

#generate predictions on the training set 
y_train_pred = model.predict(x_train_poly)
y_test_pred = model.predict(x_test_poly)

#visualise data
import plotly.graph_objects as go 

#scatter plot for training data 
trace_train = go.Scatter(x=x_train.flatten(),y=y_train.flatten(),mode='markers',name='training data',marker=dict(color='blue'))

#scatter plot for testing data 
trace_test = go.Scatter(x=x_test.flatten(),y=y_test.flatten(),mode='markers',name='testing data',marker=dict(color='green'))

#line plot for polynomial regression 
trace_regression = go.Scatter(x=x_train.flatten(),y=y_train_pred.flatten(),mode='lines',name = 'Polynomial Regression', line = dict(color='red',width=2))

#create the layout for the plot 
layout = go.Layout(title='Polynomial Regression',xaxis=dict(title='x'),yaxis=dict(title='y'))

#combine the traces and the layout and create the figure 
figure = go.Figure(data=[trace_train, trace_test, trace_regression], layout=layout)

#show the plot 
figure.show()