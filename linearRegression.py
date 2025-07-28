import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()

diabetes_x=diabetes.data[:, np.newaxis,2]

diabetes_x_train=diabetes_x[:-30]
diabetes_x_test= diabetes_x[-20:]

diabetes_y_train=diabetes.target[:-30]
diabetes_y_test=diabetes_x[-20:]

model=linear_model.LinearRegression()

model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predict=model.predict(diabetes_x_test)

print ("mean_squared_error is: ",mean_squared_error(diabetes_y_test,diabetes_y_predict))

print ("weights: ",model.coef_)
print ("intercept: ",model.intercept_)

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predict)
plt.show()
