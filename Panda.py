import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor

# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)

#calculating mean absolute error
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y,predicted_home_prices)


from sklearn.model_selection import train_test_split
#split data intp training and validation,for both features and target
#the split is based on random number generator.supplying a numeric value to
#the random_state argument gurantees we get the asme split everytime we run the script

train_X,val_X,trian_y,val_y=train_test_split(X,y, random_state = 0)
#define model
melbourne_model =DecisionTreeRegressor()
#fit model
melbourne_model.fit(train_X,train_y)
#get predicted price on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))