import joblib
import util
import pandas as pd

model = joblib.load('model_data/cur_model.pkl')
Age_scaler = joblib.load('model_data/Age_scaler.pkl')
Height_scaler = joblib.load('model_data/Height_scaler.pkl')
Sex_scaler = joblib.load('model_data/Sex_scaler.pkl')
sex_binarizer = joblib.load('model_data/sex_binarizer.pkl')


user_age = int(input('Enter your age: '))
user_height = util.in_to_cm(int(input('Enter your height in inches: ')))
user_sex = input("Are you a [M]ale or [F]emale: ").upper()

data = pd.DataFrame({'Age': [user_age], 'Height': [user_height], 'Sex': [user_sex]})


def scaler_transform(col):
    data[[col]] = eval('%s_scaler' % col).transform(data[[col]])


data['Sex'] = sex_binarizer.transform(data['Sex'])
scaler_transform('Age')
scaler_transform('Height')
scaler_transform('Sex')

user_predicted_weight = model.predict(data)[0]

print('The predicted ideal athlete at your height and weight weighs %f kgs, %f lbs'
      % (user_predicted_weight, util.kg_to_lb(user_predicted_weight)))
