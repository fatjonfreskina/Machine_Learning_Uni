import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

####PROCESSING DATA####

data_path = 'Lab0/Intro_Lab/data/km_year_power_price.csv'

df = pd.read_csv (r'Lab0/Intro_Lab/data/km_year_power_price.csv')

#For semplicity, choose one variable for the Least squares

X = pd.DataFrame(df, columns = ['km'])
X = X.to_numpy()
Y = pd.DataFrame(df, columns = ['avgPrice'])
Y = Y.to_numpy()

x_mean = X.mean()
y_mean = Y.mean()

###BUILDING THE MODEL

numerator = 0
denominator = 0

for i in range(len(X)):
	numerator += (X[i] - x_mean)*(Y[i] - y_mean)
	denominator += (X[i] - x_mean)**2

slope = numerator/denominator
intercept = y_mean - slope*x_mean

###PLOT EVERYTHING

Y_prediction = slope * X + intercept


plt.scatter(X, Y)
plt.plot(X,Y_prediction, color='red')
plt.xlabel('Kilometers')
plt.ylabel('Average Price')
plt.show()
