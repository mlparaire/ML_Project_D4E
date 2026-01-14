import json
import os
import pandas as pd
from typing import List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RepeatedStratifiedKFold, cross_val_score,train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, recall_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from scipy.sparse import hstack
from catboost import CatBoostClassifier, Pool
from multiprocessing import Pool


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
        self.main = self.main.loc[:1000]
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

    def Col_var_position(self,attribute:str):
        cat_cols = [c for c in getattr(self,attribute).columns if (getattr(self,attribute)[c].dtype == "object") or str(getattr(self,attribute)[c].dtype).startswith("string")]

        # 2) some code columns to force as categorical
        force_cat = ["INSEE", "DEP", "Reg", "job_dep", "JOB_SECURITY", "Sports"]
        for c in force_cat:
            if c in getattr(self,attribute).columns and c not in cat_cols:
                cat_cols.append(c)

        # CatBoost requires string type + fillna
        cat_idx = [getattr(self,attribute).columns.get_loc(c) for c in cat_cols]
        return cat_cols, cat_idx

    @check_if_train
    def get_train_split(self,df,y):
        return train_test_split(
        df, y, test_size=0.2, random_state=66, stratify=y
        )

    def Descision_tree_searches(self,attributes:List[str]):
        param_grid = []
        for lr in [5,6]:
            for depth in [5]:#range(5,11):
                for iter in [85]:#[85, 100, 200]
                    for leaf in [85]: #, 100, 200]:
                        param_grid.append({"learning_rate":lr,"max_depth": depth, "max_iter": iter, "min_samples_leaf": leaf})
        for attribute in attributes:
            cv_params = {'params': [], 'Accuracy': [], 'F1': [],'recall_score':[], 'score':[]} # 'confusion_matrix': []}
            X = self.One_Hot_Encode(attribute)
            print("start")
            for params in param_grid:
                print(params)
                accuracy_scores,f1_scores,recall_score_scores,scores= [],[],[],[]
                X_train, X_test, y_train, y_test = train_test_split(X.toarray(), getattr(self,attribute+'_target').to_numpy(), test_size=0.2, random_state=42)
                cv = RepeatedStratifiedKFold(n_splits=10, random_state=66)
#            cv.fit(X.toarray(), getattr(self, attribute + '_target').to_numpy())
                for train_index, val_index in cv.split(X_train, y_train):
                    accuracy_scores_reduced, f1_scores_reduced,recall_score_reduced,scores_r = [],[],[],[]
                    X_train_fold, X_val_fold,y_train_fold,y_val_fold = X_train[train_index], X_train[val_index],y_train[train_index], y_train[val_index]
                    HGB = HistGradientBoostingClassifier(
                        random_state=0,
                        **params)
                    HGB.fit(X_train_fold, y_train_fold)

                    y_pred = HGB.predict(X_val_fold)

                    accuracy_scores_reduced.append(accuracy_score(y_val_fold, y_pred))
                    f1_scores_reduced.append(f1_score(y_val_fold, y_pred,pos_label='S'))
                    recall_score_reduced.append(recall_score(y_val_fold, y_pred,pos_label='S'))
                    scores_r.append(HGB.score(X_test,y_test))
                accuracy_scores.append(np.mean(accuracy_scores_reduced))
                f1_scores.append(np.mean(f1_scores_reduced))
                recall_score_scores.append(np.mean(recall_score_reduced))
                scores.append(np.mean(scores_r))
                cv_params['params'].append(params)
                cv_params['Accuracy'].append(accuracy_scores[-1])
                cv_params['F1'].append(f1_scores[-1])
                cv_params['recall_score'].append(recall_score_scores[-1])
                cv_params['score'].append(scores[-1])

#                cv_params['confusion_matrix'].append(confusion_matrix(y_val_fold,y_pred,labels=['S','O']))

            print(accuracy_scores)
            print(f1_scores)
            best_params = {'Best Accuracy' : param_grid[np.argmax(accuracy_scores)],'Best F1' :param_grid[np.argmax(f1_scores)],
                       'Best Recall': param_grid[np.argmax(recall_score_scores)],
                       'Best Score':param_grid[np.argmax(scores)]}

            json.dump(best_params,open(f'./results/HGB/best_{attribute}_params.json','w'))

            json.dump(cv_params,open(f'./results/HGB/{attribute}_params.json','w'))
        yield attribute, cv_params,param_grid[np.argmax(accuracy_scores)],param_grid[np.argmax(f1_scores)],param_grid[np.argmax(recall_score)],param_grid[np.argmax(scores)]

    def Gradient_Boosting_search(self,attributes:List[str]):
        print('Starting Gradient_Boosting')
        cv_params = {'params': [], 'Accuracy': [], 'F1': [], 'recall_score': [],'logloss':[]}#, 'confusion_matrix': []}
        for attribute in attributes:
            param_grid = []
            for depth in [6, 8]:
                for lr in [0.05, 0.1]:
                    for l2 in [3, 7]:
                        param_grid.append({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2})
            classes = sorted(getattr(self,attribute + '_target').astype(str).unique().tolist())
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=66)
            cat_cols,cat_idx =  self.Col_var_position(attribute)

            print('Starting grid')
            for params in param_grid:
                cat_importances = np.array([])
                print(params)
                accs, f1s, lls,recalls = [], [], [],[]

                for tr, va in cv.split(getattr(self,attribute), getattr(self,attribute + '_target')):
#                    print(tr,va)
                    accus_r,f1s_r, lls_r,recall_r = [],[],[],[]
                    train_pool = Pool(getattr(self,attribute).iloc[tr].to_numpy(), getattr(self,attribute + '_target').iloc[tr].to_numpy(), cat_features=cat_idx)
                    valid_pool = Pool(getattr(self,attribute).iloc[va].to_numpy(), getattr(self,attribute + '_target').iloc[va].to_numpy(), cat_features=cat_idx)

                    model = CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=500,
                    od_type="Iter",  # early stopping
                    od_wait=50,
                    random_seed=66,
                    thread_count=-1,
                    verbose=False,
                    **params
                )
                    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

                    if len(cat_importances)== 0:
                        cat_importances = model.feature_importances_
                    else:
                        cat_importances = np.vstack([cat_importances,model.feature_importances_])

                    pred = model.predict(valid_pool).astype(str).ravel()
                    proba = model.predict_proba(valid_pool)

                    y_true = getattr(self,attribute + '_target').iloc[va].astype(str).values
                    accus_r.append(accuracy_score(y_true, pred))
                    f1s_r.append(f1_score(y_true, pred, average="macro"))
                    lls_r.append(log_loss(y_true, proba, labels=classes))
                    recall_r.append(recall_score(y_true, pred,pos_label='S'))
                accs.append(np.mean(accus_r))
                f1s.append(np.mean(f1s_r))
                lls.append(np.mean(lls_r))
                recalls.append(np.mean(recall_r))
                cv_params['params'].append(params)
                cv_params['Accuracy'].append(accs[-1])
                cv_params['logloss'].append(lls[-1])
                cv_params['F1'].append(f1s[-1])
                cv_params['recall_score'].append(recalls[-1])
#                cv_params['confusion_matrix'].append(confusion_matrix(y_true, pred, labels=['S', 'O']))

            best_params = {'Best Accuracy' : param_grid[np.argmax(accs)],'Best F1' :param_grid[np.argmax(f1s)],
                       'Best Recall': param_grid[np.argmax(recalls)]}

            json.dump(best_params,open(f'./results/catboost/best_{attribute}_params.json','w'))

            json.dump(cv_params,open(f'./results/catboost/{attribute}_params.json','w'))
            pd.DataFrame(cat_importances,columns=[col for col in getattr(self,attribute).columns]).to_csv(f'./results/catboost/{attribute}_varibales.csv',sep=";",index=False)

        yield attribute, cv_params, param_grid[np.argmax(accs)], param_grid[np.argmax(f1s)], param_grid[np.argmax(recalls)]

    def Random_ForestClassifier(self,attributes:List[str]):
        print('Starting RandomForest_classifier')
        cv_params = {'params': [], 'Accuracy': [], 'F1': [], 'recall_score': [],
                     'score': []}  # , 'confusion_matrix': []}
        param_grid = []
        for depth in [6, 8]:
            for lr in [6,8]:
                for l2 in [3, 7]:
                    param_grid.append({"max_depth": depth, "max_leaf_nodes": lr, "max_samples": l2})
        for attribute in attributes:
            X = self.One_Hot_Encode(attribute)
            print("start")
            for params in param_grid:
                print(params)
                accuracy_scores,f1_scores,recall_score_scores,scores= [],[],[],[]
                X_train, X_test, y_train, y_test = train_test_split(X.toarray(), getattr(self,attribute+'_target').to_numpy(), test_size=0.2, random_state=42)
                cv = RepeatedStratifiedKFold(n_splits=10, random_state=66)
#            cv.fit(X.toarray(), getattr(self, attribute + '_target').to_numpy())
                for train_index, val_index in cv.split(X_train, y_train):
                    accuracy_scores_reduced, f1_scores_reduced,recall_score_reduced,scores_r = [],[],[],[]
                    X_train_fold, X_val_fold,y_train_fold,y_val_fold = X_train[train_index], X_train[val_index],y_train[train_index], y_train[val_index]
                    RFC = RandomForestClassifier(
                        random_state=0,
                        **params)
                    RFC.fit(X_train_fold, y_train_fold)

                    y_pred = RFC.predict(X_val_fold)

                    accuracy_scores_reduced.append(accuracy_score(y_val_fold, y_pred))
                    f1_scores_reduced.append(f1_score(y_val_fold, y_pred,pos_label='S'))
                    recall_score_reduced.append(recall_score(y_val_fold, y_pred,pos_label='S'))
                    scores_r.append(RFC.score(X_test,y_test))
                accuracy_scores.append(np.mean(accuracy_scores_reduced))
                f1_scores.append(np.mean(f1_scores_reduced))
                recall_score_scores.append(np.mean(recall_score_reduced))
                scores.append(np.mean(scores_r))
                cv_params['params'].append(params)
                cv_params['Accuracy'].append(accuracy_scores[-1])
                cv_params['F1'].append(f1_scores[-1])
                cv_params['recall_score'].append(recall_score_scores[-1])
                cv_params['score'].append(scores[-1])

#                cv_params['confusion_matrix'].append(confusion_matrix(y_val_fold,y_pred,labels=['S','O']))

            print(accuracy_scores)
            print(f1_scores)
            best_params = {'Best Accuracy' : param_grid[np.argmax(accuracy_scores)],'Best F1' :param_grid[np.argmax(f1_scores)],
                       'Best Recall': param_grid[np.argmax(recall_score_scores)],
                       'Best Score':param_grid[np.argmax(scores)]}

            json.dump(best_params,open(f'./results/RFC/best_{attribute}_params.json','w'))

            json.dump(cv_params,open(f'./results/RFC/{attribute}_params.json','w'))
        yield attribute, cv_params,param_grid[np.argmax(accuracy_scores)],param_grid[np.argmax(f1_scores)],param_grid[np.argmax(recall_score)],param_grid[np.argmax(scores)]

    @check_if_train
    def find_best_model(self,result_RFC,result_grad,result_HGB,attributes:List[str]):
        for attribute in attributes:
            for item in result_RFC:
                if result_RFC[0] == attribute:
                    print(result_RFC[1])


## Prediction
learning = initial_data(learning_vars,is_train=True)
#testing = initial_data(test_vars,is_train=False)

#learning.preprocessing()
print('Processing testing dataset')
result_HGB = learning.Descision_tree_searches(['main','student'])
result_grad = learning.Gradient_Boosting_search(['main','student'])
result_HGB = learning.Descision_tree_searches(['main','student'])

learning.find_best_model(['main'])