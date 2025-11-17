import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error  


# ici on va benchmark ma classe de regression linéaire , et celle de scikitlearn 
from sklearn.linear_model import LinearRegression 
from LR import LinearRegression as LR_custom


# je sais que j'ai deja fait une competition kaggle , housing prices, ou le model de regression lineaire est coherent

X = pd.read_csv('train.csv')
features_avec_y = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars', 'SalePrice']
X = X[features_avec_y]
X = X.dropna()



y = X['SalePrice']
X = X[['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_time1 = time.perf_counter()
model1 = LinearRegression()
model1.fit(X_train_scaled, y_train)
end_time1 = time.perf_counter  ()

start_time2 = time.perf_counter()
model2 = LR_custom(learning_rate=0.01, n_iterations=1000) 
model2.fit(X_train_scaled, y_train.values )
end_time2 = time.perf_counter() 

print(f"temps de fit de scikit learn : {(end_time1-start_time1) :.2f}s")
print(f"temps de fit de mon model linéaire avec 1000 itérations : {(end_time2-start_time2)}s")

y1 = model1.predict(X_test_scaled)
y2 = model2.predict(X_test_scaled)

rmse1 = np.sqrt(mean_squared_error(y_test, y1))
rmse2 = np.sqrt(mean_squared_error(y_test, y2)) 

print(f"RMSE Modèle Sklearn : {rmse1:.2f}")
print(f"RMSE Modèle Custom  : {rmse2:.2f}") 








