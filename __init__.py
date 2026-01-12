import json
import os
import pandas as pd
from typing import List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import hstack



test_vars = json.load(open('./catboost_params/test_vars.json',"r"))
learning_vars = json.load(open('./catboost_params/learning_vars.json',"r"))
geography = json.load(open('./catboost_params/geography.json',"r"))

def make_path(begin:str, end:str) -> str:
    return os.path.join(os.path.abspath(begin), end)

class initial_data:
    def __init__(self,dataset:pd.DataFrame,is_train:bool):
        self.is_train = is_train
        self.has_been_called = True
        self.encoder = OneHotEncoder(handle_unknown='ignore',dtype=np.float16)
        self.data_dir : str = os.path.abspath('project-13-files')

        self.dt = HistGradientBoostingClassifier(random_state=0)

        for var,item in dataset.items():
            setattr(self,var,pd.read_csv(make_path(self.data_dir,item)))
        for var,item in geography.items():
            setattr(self,var,pd.read_csv(make_path(self.data_dir,item)))

        for var in list(dataset.keys())[1:]:
            self.main = self.main.merge(getattr(self, var), on='UID', how='left')
        for geog in geography.keys():
            self.main = self.main.merge(getattr(self, geog), on='INSEE', how='left')
        ## For categorical variables
        for dest, cat in dict({'is_retiree':'retired_jobs','is_sport_member':"sport"}).items():
            self.main[dest] = self.main['UID'].isin(getattr(self,cat)['UID'])

        for test in self.main.iloc[:,[x.endswith('_x') for x in self.main.columns]]:
            self.main.loc[self.main[test].apply(pd.isna),test] = self.main.loc[self.main[test].apply(pd.isna),test.replace('_x','_y')]

        self.main = self.main.iloc[:,[not x.endswith('_y') for x in self.main.columns]]
        self.main.columns = [x.replace('_x','') for x in self.main.columns]
        self.main=self.main.drop(['X','Y','LAT','Long'],axis=1)
    def __len__(self):
        return len(self.main)

    def check_if_train(func):
        def wrapper(self):
            if self.is_train:
                func(self)
            else:
                raise 'Impossible on test Dataset'
        return wrapper

    def preprocessing(self):
        self.enc = OneHotEncoder(handle_unknown='ignore')
        if self.is_train:
            self.y = self.main[["target",'UID']].astype("string").fillna("__MISSING_TARGET__")
        else:
            self.UID = self.main["UID"].copy()

        self.__string_vars =  list(self.main.dtypes == 'object')

        self.main.loc[:,self.__string_vars] = self.main.loc[:,self.__string_vars].fillna('Missing').astype(str)

        self.unemployed = self.main[(self.main.employee_count == "Missing") & (~self.main.is_retiree) & (~self.main.Is_student)]
        self.student = self.main[(self.main.employee_count == "Missing") & (~self.main.Is_student)]
        self.retiree = self.main[(self.main.is_retiree)]

        self.student_UID = self.student.UID
        self.unemployed_UID = self.unemployed.UID
        self.retiree_UID = self.retiree.UID
        self.main_UID = self.main.UID

        self.main = self.main.drop(self.unemployed.index,errors = "ignore")
        self.main = self.main.drop(self.retiree.index,errors = "ignore")
        self.main = self.main.drop(self.student.index,errors = "ignore")

        self.unemployed.drop(columns=['UID','Is_student','retirement_age','is_retiree'],inplace=True)
        self.retiree.drop(columns=['UID','is_retiree','Is_student'],inplace=True)
        self.student.drop(columns=['UID','Is_student','retirement_age','is_retiree'],inplace=True)
        self.main.drop(columns=['UID','Is_student','retirement_age','is_retiree',],inplace=True)

        self.main_target = self.main['target']
        self.main.drop(columns=["target"], inplace=True)

        self.unemployed_target = self.unemployed['target']
        self.unemployed.drop(columns=["target"], inplace=True)

        self.retiree_target = self.retiree['target']
        self.retiree.drop(columns=["target"], inplace=True)

        self.student_target = self.student['target']
        self.student.drop(columns=["target"], inplace=True)

        print('Merging Dataset')

    def One_Hot_Encode(self,attribute: str):
        return hstack([getattr(self,attribute).select_dtypes(exclude="object").to_numpy(dtype=np.float32), self.enc.fit_transform(getattr(self,attribute).select_dtypes(include="object"))])

    def Descision_tree_searches(self,attributes:List[str]):
        v_split = StratifiedShuffleSplit(n_splits=1, train_size=0.7)
        param_grid = {'max_depth': range(1, 11),
                      'max_iter': [2, 10, 25, 50, 75, 85, 100, 200],
                      "min_samples_leaf": [2, 10, 25, 50, 75, 85, 100, 200]}
        for attribute in attributes:
            X = self.One_Hot_Encode(attribute)
            dt_search = GridSearchCV(self.dt, param_grid, cv=v_split, n_jobs=-1)
            dt_res = dt_search.fit(X.toarray(), getattr(self, attribute+'_target'))
            return dt_res

    @check_if_train
    def get_train_split(self,df,y):
        return train_test_split(
        df, y, test_size=0.2, random_state=66, stratify=y
        )

## Prediction
learning = initial_data(learning_vars,is_train=True)
#testing = initial_data(test_vars,is_train=False)

learning.preprocessing()
print('Processing testing dataset')
print(learning.Descision_tree_searches(['main']))