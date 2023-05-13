import pandas as p
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Gathering data
data = p.read_csv( 'C:/Users/AWIKSSHIITH/OneDrive/Desktop/housing.csv/housing.csv' )
X = np.array( data[ 'total_rooms' ] )
Y = np.array( data[ 'median_house_value' ] )
X = X.reshape( len(X), 1 ) #Reshaping the vertical array to horizontal one
Y = X.reshape( len(Y), 1 ) #Reshaping the vertical array to horizontal one

#Splitting the data
X_train = X[ : 14447 ]
X_test = X[ 14447 : ]
Y_train = Y[ : 14447 ]
Y_test = Y[ 14447 : ]

#Creating regression curve, training & testing the machine
regr = linear_model.LinearRegression()
regr.fit( X_train, Y_train ) #Training the machine
plt.scatter( X_test, Y_test , color='green' )
plt.plot( X_test, regr.predict( X_test ), color = 'red', linewidth = 3 ) #Testing the machine and creating a regression curve
plt.title( 'Price of House in California State with respect to number of rooms' )
plt.xlabel( 'Number of Rooms' )
plt.ylabel( 'Price of House' )
plt.show() # Comparing the output value given by the machine and the actual output

#Prediction for non pre_defined output
print("Prediction of cost for a 2 room house:")
print( regr.predict( 2 ) )