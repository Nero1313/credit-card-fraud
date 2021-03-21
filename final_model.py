import numpy as np
import pandas as pd

data = pd.read_csv('./data/train.csv')

data = data.drop(['Time'], axis=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
X = scaler.fit_transform(X)

# print('Shape of X: {}'.format(X.shape))
# print('Shape of y: {}'.format(y.shape))

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X, y.ravel())

# from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# rf1= RandomForestClassifier(random_state = 100, n_estimators = 100)
#
# param_grid = {'n_estimators':[10,25,100],
#          'min_samples_split':[3,5,10],
#        'class_weight':['balanced',None],
#         'max_depth':[3,5,None]}
#
# grid = GridSearchCV(rf1,param_grid,n_jobs=-1,verbose=1,cv=2,scoring='f1')

# grid.fit(X,y)
# grid.best_score_
# grid.best_estimator_.get_params()

rf2 = RandomForestClassifier(random_state = 100, class_weight= None,min_samples_split= 5,
                             n_estimators= 100)

rf2.fit(X,y)

import joblib
filename = 'finalized_model.sav'
joblib.dump(rf2, filename)