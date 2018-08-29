import joblib
import util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load('model_data/cur_model.pkl')
Age_scaler = joblib.load('model_data/Age_scaler.pkl')
Height_scaler = joblib.load('model_data/Height_scaler.pkl')
Sex_scaler = joblib.load('model_data/Sex_scaler.pkl')
sex_binarizer = joblib.load('model_data/sex_binarizer.pkl')


# user_age = int(input('Enter your age: '))
# user_height = util.in_to_cm(int(input('Enter your height in inches: ')))
# user_sex = input("Are you a [M]ale or [F]emale: ").upper()
num_elem = 200
age_range = np.linspace(16, 40, num_elem)
data = pd.DataFrame({'Age': age_range, 'Height': [178] * num_elem, 'Sex': ['M'] * num_elem})


def scaler_transform(col):
    data[[col]] = eval('%s_scaler' % col).transform(data[[col]])


data['Sex'] = sex_binarizer.transform(data['Sex'])
scaler_transform('Age')
scaler_transform('Height')
scaler_transform('Sex')

user_predicted_weights = model.predict(data)

# print('The predicted ideal athlete at your height and weight weighs %f kgs, %f lbs'
#       % (user_predicted_weight, util.kg_to_lb(user_predicted_weight)))
print(user_predicted_weights)

plt.plot(age_range, [util.kg_to_lb(user_predicted_weights[i]) for i in range(len(user_predicted_weights))])

plt.show()
