import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle
df = pd.read_csv('ipl.csv')

X = df[['6th_run','1st_wicket','14th_over','2nd_wicket']]
Y = df.final_score
X_train , X_test , y_train , y_test = train_test_split(X, Y , test_size = 0.25 , random_state= 20)

model = LinearRegression()
model.fit(X_train, y_train)
filename = 'ipl_pred.sav'
pickle.dump(model, open(filename, 'wb'))


