import scipy as sp
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
import joblib


# data preprocessing
olympic_data = pd.read_csv('athlete_events.csv')

olympic_data = olympic_data.drop('Medal', axis=1)  # medal contained many missing values (any time someone didn't place)
olympic_data = olympic_data.dropna(axis=0)  # drop missing values

predictors = ['Age', 'Height', 'Sex']
response = 'Weight'

# Change Sex column from categorical so it fits the model
sex_binarizer = LabelBinarizer()
sex_binarizer.fit(olympic_data['Sex'])
olympic_data['Sex'] = sex_binarizer.transform(olympic_data['Sex'])

# create a scaler for each predictor that I can save and run on data I want to predict
predictor_scalers = {predictor: StandardScaler() for predictor in predictors}
for predictor in predictor_scalers:
    predictor_scalers[predictor].fit(olympic_data[[predictor]])
    olympic_data[[predictor]] = predictor_scalers[predictor].transform(olympic_data[[predictor]])

# define X and y and split into test and training data
X = olympic_data[predictors]
y = olympic_data.Weight
test_size = 0.33
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=1)


saved_model = joblib.load('model_data/cur_model.pkl')
saved_prediction = saved_model.predict(test_X)
saved_mae = mean_absolute_error(test_y, saved_prediction)
print('mae of saved model = ', saved_mae)

model = MLPRegressor(random_state=1)
model.fit(train_X, train_y)
test_prediction = model.predict(test_X)
mae = mean_absolute_error(test_y, test_prediction)
print('mae', mae)

if input('Would you like to save this model? [y/n]').lower() == 'y':
    for predictor in predictors:
        joblib.dump(predictor_scalers[predictor], 'model_data/%s_scaler.pkl' % predictor)
    joblib.dump(sex_binarizer, 'model_data/sex_binarizer.pkl')
    joblib.dump(model, 'model_data/cur_model.pkl')







