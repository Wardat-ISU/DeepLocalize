#!/usr/bin/python

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split

#import pdb; pdb.set_trace()

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_model = DecisionTreeRegressor()
melbourne_data = melbourne_data.dropna(axis=0)


y = melbourne_data.Price
X = melbourne_data[['Rooms','Bathroom','Landsize','Bathroom','YearBuilt','Car','Distance']]

trainX, ValX, trainY, ValY = train_test_split(X,y)

melbourne_model.fit(trainX,trainY)

print("Making predictions for following houses")
predictions = melbourne_model.predict(ValX)

print("MAE:",mean_absolute_error(predictions,ValY))

# The Melbourne data has somemissing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
#Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
