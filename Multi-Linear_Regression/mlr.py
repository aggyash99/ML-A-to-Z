import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('50_Startups.csv')
# print (data)
x= data.iloc[:,:-1].values
# print(x)
y=data.iloc[:,4].values

"""from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()
"""labelencoder_y=LabelEncoder()
y=labelencoder_x.fit_transform(y)"""

#Avoiding dummy var. trap
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#scaling is automatic
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#with all dep. var.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

#optimal model with back. el.
#it does not account of b0 variable in eq.
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regerssor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regerssor_ols.summary()
x_opt=x[:,[0,1,3,4,5]]
regerssor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regerssor_ols.summary()
x_opt=x[:,[0,3,4,5]]
regerssor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regerssor_ols.summary()
x_opt=x[:,[0,3,5]]
regerssor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regerssor_ols.summary()
x_opt=x[:,[0,3]]
regerssor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regerssor_ols.summary()
