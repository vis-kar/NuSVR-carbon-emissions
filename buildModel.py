from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

df = pd.read_csv('./carbon_emissions.csv')
df = df.drop(['ID'], axis=1) # del df['ID']

X = df.drop('carbon_emissions_metric_ton', axis=1)
y = df['carbon_emissions_metric_ton']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = NuSVR(nu=0.5, kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared score:', r2)

import pickle

with open('./NuSVR.pkl', 'wb') as f:
    pickle.dump(model, f)