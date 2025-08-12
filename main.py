import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import joblib


Base_dir = os.path.dirname(__file__)
data_path = os.path.join(Base_dir, "Data")
model_path = os.path.join(Base_dir, "Model")
Csv_path = os.path.join(data_path, "WineQT.csv")

data = pd.read_csv(Csv_path)
data.drop(columns=['Id'], inplace=True)

# print(data.head(5))
# print(data.isnull().sum())
# print(data.dtypes)
# print(data.info())
# print(data.describe())


x=data.drop(columns=['quality'])
y=data['quality']

scaler=MinMaxScaler()
x=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)

params={
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}

model  =GridSearchCV(SVC(),params,cv=5,scoring='accuracy',n_jobs=-1,verbose=2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("SVC Accuracy:", accuracy_score(y_test, y_pred))
print("SVC Precision:", precision_score(y_test, y_pred, average='weighted'))


joblib.dump(model, os.path.join(model_path, "svc_model.pkl"))