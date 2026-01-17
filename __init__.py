import json
import os
import pandas as pd
from typing import List,Dict
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
from numpy import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
import seaborn as sns


from matplotlib import pyplot as plt
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

    def __len__(self):
        return len(self.main)

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

        if self.is_train:
            self.main_target = self.main['target']
            self.main.drop(columns=["target"], inplace=True)

            self.unemployed_target = self.unemployed['target']
            self.unemployed.drop(columns=["target"], inplace=True)

            self.retiree_target = self.retiree['target']
            self.retiree.drop(columns=["target"], inplace=True)

            self.student_target = self.student['target']
            self.student.drop(columns=["target"], inplace=True)

        print('Preprocessing done')

    def Prepare_XGBData(self, attribute: str):
        df = getattr(self, attribute).copy()

        cat_cols = df.select_dtypes(include="object").columns.tolist()

        for col in cat_cols:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map)

        return df.to_numpy(dtype=np.float32)

    def One_Hot_Encode(self,attribute: str):
        return hstack([getattr(self,attribute).select_dtypes(exclude="object").to_numpy(dtype=np.float32), self.enc.fit_transform(getattr(self,attribute).select_dtypes(include="object")).toarray()])
    def Col_var_position(self,attribute:str):
        cat_cols = [c for c in getattr(self,attribute).columns if (getattr(self,attribute)[c].dtype == "object") or str(getattr(self,attribute)[c].dtype).startswith("string")]

        force_cat = ["INSEE", "DEP", "Reg", "job_dep", "JOB_SECURITY", "Sports"]
        for c in force_cat:
            if c in getattr(self,attribute).columns and c not in cat_cols:
                cat_cols.append(c)

        # CatBoost requires string type + fillna
        cat_idx = [getattr(self,attribute).columns.get_loc(c) for c in cat_cols]
        return cat_cols, cat_idx

    def get_train_split(self,df,y):
        return train_test_split(
        df.to_numpy(), y.to_numpy(), test_size=0.2, random_state=66, stratify=y
        )
    def Gradient_Boosting_search(self,attributes:List[str]):
        print('Starting Gradient_Boosting')
        cv_params = {'params': [], 'Accuracy': [], 'F1': [], 'recall_score': [],'logloss':[]}#, 'confusion_matrix': []}
        for attribute in attributes:
            param_grid = []
            for depth in [5, 6, 8]:
                for lr in [0.05, 0.1]:
                    for l2 in [3, 7]:
                        param_grid.append({
                            "depth": depth,
                            "learning_rate": lr,
                            "l2_leaf_reg": l2
                        })
            classes = sorted(getattr(self,attribute + '_target').astype(str).unique().tolist())
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=66)
            cat_cols,cat_idx =  self.Col_var_position(attribute)
            print(f"starting CatBoostClassifier for {attribute}")
            accs, f1s, lls, recalls = [], [], [], []
            for params in param_grid:
                cat_importances = np.array([])
                print(params)
                accus_r, f1s_r, lls_r, recall_r = [], [], [], []
                X_tr, X_va, y_tr, y_va = learning.get_train_split(getattr(self,attribute),getattr(self,attribute + '_target'))
                for tr, va in cv.split(getattr(self,attribute), getattr(self,attribute + '_target')):
#                    print(tr,va)
                    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
                    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

                    model = CatBoostClassifier(
                        loss_function="Logloss",
                        iterations=300,
                        subsample= 0.8,
                        od_type="Iter",  # early stopping
                        od_wait=20,
                        max_bin=128,
                        rsm = 0.8,
                        sampling_frequency="PerTree",
                        grow_policy="SymmetricTree",
                        random_seed=66,
                        thread_count=-1,
                        verbose=False,
                        **params
                )
                    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

                    pred = model.predict(valid_pool).astype(str).ravel()
                    proba = model.predict_proba(valid_pool)

                    y_true = y_va
                    accus_r.append(accuracy_score(y_true, pred))
                    f1s_r.append(f1_score(y_true, pred, average="macro"))
                    lls_r.append(log_loss(y_true, proba, labels=classes))
                    recall_r.append(recall_score(y_true, pred,pos_label='S'))
                    if len(cat_importances) == 0:
                        cat_importances = model.feature_importances_
                    else:
                        cat_importances = np.vstack([cat_importances, model.feature_importances_])
                accs.append(np.mean(accus_r))
                f1s.append(np.mean(f1s_r))
                lls.append(np.mean(lls_r))
                recalls.append(np.mean(recall_r))
                cv_params['params'].append(params)
                cv_params['Accuracy'].append(accs[-1])
                cv_params['logloss'].append(lls[-1])
                cv_params['F1'].append(f1s[-1])
                cv_params['recall_score'].append(recalls[-1])
#                plt.imshow(confusion_matrix(y_true, pred, labels=['S', 'O']))
#                plt.imsave(f'./img/confusion_matrix/gradboos/grad_boosting_{params}.png')
            best_params = {'Best Accuracy' : param_grid[np.argmax(accs)],'Best F1' :param_grid[np.argmax(f1s)],
                       'Best Recall': param_grid[np.argmax(recalls)]}

            json.dump(best_params,open(f'./results/catboost/best_{attribute}_params.json','w'))

            json.dump(cv_params,open(f'./results/catboost/{attribute}_params.json','w'))
            pd.DataFrame(cat_importances,columns=[col for col in getattr(self,attribute).columns]).to_csv(f'./results/catboost/{attribute}_variables.csv',sep=";",index=False)

        yield attribute, cv_params, param_grid[np.argmax(accs)], param_grid[np.argmax(f1s)], param_grid[np.argmax(recalls)]

    def Descision_tree_searches(self,attributes:List[str]):
        param_grid = []
        for lr in [6,7,8]:
            for depth in range(5,11):#range(5,11):
                for iter in [85, 100, 200]:
                    for leaf in [20, 50, 100]:
                        param_grid.append({"learning_rate":lr,"max_depth": depth, "max_iter": iter, "min_samples_leaf": leaf})
                for attribute in attributes:
                    print(attribute)
                    cv_params = {'params': [], 'Accuracy': [], 'F1': [], 'recall_score': [],
                                 'score': []}  # 'confusion_matrix': []}
                    X = self.One_Hot_Encode(attribute)
                    print("start")
                    for params in param_grid:
                        print(params)
                        accuracy_scores, f1_scores, recall_score_scores, scores = [], [], [], []
                        X_train, X_test, y_train, y_test = train_test_split(X.toarray(), getattr(self,
                                                                                                 attribute + '_target').to_numpy(),
                                                                            test_size=0.2, random_state=42)
                        cv = StratifiedKFold(n_splits=10, random_state=66)
                        for train_index, val_index in cv.split(X_train, y_train):
                            accuracy_scores_reduced, f1_scores_reduced, recall_score_reduced, scores_r = [], [], [], []
                            X_train_fold, X_val_fold, y_train_fold, y_val_fold = X_train[train_index], X_train[
                                val_index], y_train[train_index], y_train[val_index]
                            HGB = HistGradientBoostingClassifier(
                                random_state=66,
                                max_iter=500,
                                early_stopping=True,
                                validation_fraction=0.1,
                                **params)
                            HGB.fit(X_train_fold, y_train_fold)
                            y_pred = HGB.predict(X_val_fold)

                            accuracy_scores_reduced.append(accuracy_score(y_val_fold, y_pred))
                            f1_scores_reduced.append(f1_score(y_val_fold, y_pred, pos_label='S'))
                            recall_score_reduced.append(recall_score(y_val_fold, y_pred, pos_label='S'))
                            scores_r.append(HGB.score(X_test, y_test))
                        accuracy_scores.append(np.mean(accuracy_scores_reduced))
                        f1_scores.append(np.mean(f1_scores_reduced))
                        recall_score_scores.append(np.mean(recall_score_reduced))
                        scores.append(np.mean(scores_r))
                        cv_params['params'].append(params)
                        cv_params['Accuracy'].append(accuracy_scores[-1])
                        cv_params['F1'].append(f1_scores[-1])
                        cv_params['recall_score'].append(recall_score_scores[-1])
                        cv_params['score'].append(scores[-1])
            print(accuracy_scores)
            print(f1_scores)
            best_params = {'Best Accuracy' : param_grid[np.argmax(accuracy_scores)],'Best F1' :param_grid[np.argmax(f1_scores)],
                       'Best Recall': param_grid[np.argmax(recall_score_scores)],
                       'Best Score':param_grid[np.argmax(scores)]}

            json.dump(best_params,open(f'./results/HGB/best_{attribute}_params.json','w'))

            json.dump(cv_params,open(f'./results/HGB/{attribute}_params.json','w'))
        yield attribute, cv_params,param_grid[np.argmax(accuracy_scores)],param_grid[np.argmax(f1_scores)],param_grid[np.argmax(recall_score)],param_grid[np.argmax(scores)]


    def Random_ForestClassifier(self,attributes:List[str]):
        print('Starting RandomForest_classifier')
        cv_params = {'params': [], 'Accuracy': [], 'F1': [], 'recall_score': [],
                     'score': []}  # , 'confusion_matrix': []}
        param_grid = []
        for max_depth in [None, 6, 10]:
            for min_leaf in [1, 5, 20]:
                for max_features in ["sqrt", 0.5]:
                    for max_samples in [0.7, 1.0]:
                        param_grid.append({
                            "n_estimators": 300,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_features,
                            "max_samples": max_samples
                        })
                for attribute in attributes:
                    X = self.One_Hot_Encode(attribute)
                    print("start")
                    for params in param_grid:
                        print(params)
                        accuracy_scores, f1_scores, recall_score_scores, scores = [], [], [], []
                        X_train, X_test, y_train, y_test = train_test_split(X.toarray(), getattr(self,attribute + '_target').to_numpy(),test_size=0.2, random_state=42)
                        cv = StratifiedKFold(n_splits=10, random_state=66)
                        for train_index, val_index in cv.split(X_train, y_train):
                            accuracy_scores_reduced, f1_scores_reduced, recall_score_reduced, scores_r = [], [], [], []
                            X_train_fold, X_val_fold, y_train_fold, y_val_fold = X_train[train_index], X_train[
                                val_index], y_train[train_index], y_train[val_index]
                            RFC = RandomForestClassifier(
                                random_state=0,
                                **params)
                            RFC.fit(X_train_fold, y_train_fold)

                            y_pred = RFC.predict(X_val_fold)

                            accuracy_scores_reduced.append(accuracy_score(y_val_fold, y_pred))
                            f1_scores_reduced.append(f1_score(y_val_fold, y_pred, pos_label='S'))
                            recall_score_reduced.append(recall_score(y_val_fold, y_pred, pos_label='S'))
                            scores_r.append(RFC.score(X_test, y_test))
                        accuracy_scores.append(np.mean(accuracy_scores_reduced))
                        f1_scores.append(np.mean(f1_scores_reduced))
                        recall_score_scores.append(np.mean(recall_score_reduced))
                        scores.append(np.mean(scores_r))
                        cv_params['params'].append(params)
                        cv_params['Accuracy'].append(accuracy_scores[-1])
                        cv_params['F1'].append(f1_scores[-1])
                        cv_params['recall_score'].append(recall_score_scores[-1])
                        cv_params['score'].append(scores[-1])

                    print(accuracy_scores)
                    print(f1_scores)
                    best_params = {'Best Accuracy': param_grid[np.argmax(accuracy_scores)],
                                   'Best F1': param_grid[np.argmax(f1_scores)],
                                   'Best Recall': param_grid[np.argmax(recall_score_scores)],
                                   'Best Score': param_grid[np.argmax(scores)]}

                    json.dump(best_params, open(f'./results/RFC/best_{attribute}_params.json', 'w'))

                    json.dump(cv_params, open(f'./results/RFC/{attribute}_params.json', 'w'))
                yield attribute, cv_params, param_grid[np.argmax(accuracy_scores)], param_grid[np.argmax(f1_scores)],param_grid[np.argmax(recall_score)], param_grid[np.argmax(scores)]

    def XGBoosting_search(self,attributes:List[str]):
        print('Starting XGBoosting')
        cv_params = {'params': [], 'Accuracy': [], 'F1': [], 'recall_score': [],
                     'logloss': []}  # , 'confusion_matrix': []}
        for attribute in attributes:
            param_grid = []
            for depth in [6, 7, 8]:
                for lr in [0.01,0.03, 0.05]:
                    for l2 in [3, 7]:
                        param_grid.append({
                            "max_depth": depth,
                            "learning_rate": lr,
                            "reg_lambda": l2,
                        })
            y = getattr(self, attribute + "_target").astype(str).values
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            X = self.Prepare_XGBData(attribute)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=66)
            print(f"starting XGBoosting for {attribute}")
            accs, f1s, lls, recalls = [], [], [], []
            for params in param_grid:
                print(params)
                params_full = params.copy()
                params_full.update({
                    "colsample_bytree": 0.8,
                    "tree_method": "hist",
                    "n_jobs": -1,
                    "subsample": 0.8,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss"
                })
                accus_r, f1s_r, lls_r, recall_r = [], [], [], []
                for tr, va in cv.split(X,y_enc):
                    dtrain = xgb.DMatrix(X[tr], label=y_enc[tr])
                    dvalid = xgb.DMatrix(X[va], label=y_enc[va])

                    model = xgb.train(
                        params_full,
                        dtrain,
                        num_boost_round=300,
                        evals=[(dvalid, "validation")],
                        early_stopping_rounds=20,
                        verbose_eval=False
                    )
                    prob = model.predict(dvalid)
                    pred = (prob > 0.5).astype(int)
                    accus_r.append(accuracy_score(y_enc[va], pred))
                    f1s_r.append(f1_score(y_enc[va], pred, average="macro"))
                    recall_r.append(recall_score(y_enc[va], pred))
                    lls_r.append(log_loss(y_enc[va], prob))

                accs.append(np.mean(accus_r))
                f1s.append(np.mean(f1s_r))
                lls.append(np.mean(lls_r))
                recalls.append(np.mean(recall_r))
                cv_params['params'].append(params)
                cv_params['Accuracy'].append(accs[-1])
                cv_params['logloss'].append(lls[-1])
                cv_params['F1'].append(f1s[-1])
                cv_params['recall_score'].append(recalls[-1])
            #                plt.imshow(confusion_matrix(y_true, pred, labels=['S', 'O']))
            #                plt.imsave(f'./img/confusion_matrix/gradboos/grad_boosting_{params}.png')
            best_params = {'Best Accuracy': param_grid[np.argmax(accs)], 'Best F1': param_grid[np.argmax(f1s)],
                           'Best Recall': param_grid[np.argmax(recalls)]}

            json.dump(best_params, open(f'./results/XGboost/best_{attribute}_params.json', 'w'))

            json.dump(cv_params, open(f'./results/XGboost/{attribute}_params.json', 'w'))

        yield attribute, cv_params, param_grid[np.argmax(accs)], param_grid[np.argmax(f1s)], param_grid[
            np.argmax(recalls)]

    def find_best_model_performances(self,attributes: List[str], results_dir: str = './results') -> pd.DataFrame:
        algorithms = {
            'CatBoost': 'catboost',
            'XGBoosting': 'XGboost',
        }

        metrics = ['Accuracy', 'F1', 'recall_score']
        if 'CatBoost' in algorithms:
            metrics = metrics + ['logloss']

        all_results = []

        for attribute in attributes:
            print(f"\n{'=' * 60}")
            print(f"Analyzing attribute: {attribute}")
            print(f"{'=' * 60}")

            attribute_results = {'Attribute': attribute}

            for algo_name, algo_dir in algorithms.items():
                try:
                    results_file = f'{results_dir}/{algo_dir}/{attribute}_params.json'
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                    available_metrics = metrics.copy()
                    if algo_name == 'CatBoost' and 'logloss' in results:
                        available_metrics.append('logloss')

                    for metric in available_metrics:
                        if metric in results:
                            values = results[metric]
                            params_list = results['params']

                            if metric == 'logloss':
                                # For logloss, lower is better
                                best_idx = np.argmin(values)
                                best_value = values[best_idx]
                            else:
                                # For other metrics, higher is better
                                best_idx = np.argmax(values)
                                best_value = values[best_idx]

                            best_params = params_list[best_idx]

                            attribute_results[f'{algo_name}_{metric}'] = best_value
                            attribute_results[f'{algo_name}_{metric}_params'] = str(best_params)

                            print(f"\n{algo_name} - Best {metric}: {best_value:.4f}")
                            print(f"  Parameters: {best_params}")

                except FileNotFoundError:
                    print(f"\nWarning: Results file not found for {algo_name} on {attribute}")
                    for metric in metrics:
                        attribute_results[f'{algo_name}_{metric}'] = None
                        attribute_results[f'{algo_name}_{metric}_params'] = None
                except Exception as e:
                    print(f"\nError processing {algo_name} for {attribute}: {str(e)}")

            all_results.append(attribute_results)

        df_results = pd.DataFrame(all_results)

        return df_results

    def export_best_overall_parameters(self,attributes: List[str],
                                       results_dir: str = './results',
                                       output_file: str = './best_models_overall.json'):

        algorithms = {
            'CatBoost': 'catboost',
            'XGBoosting': 'XGboost',
        }

        best_models = {}

        print(f"\n{'=' * 70}")
        print("FINDING BEST OVERALL MODEL FOR EACH ATTRIBUTE")
        print(f"{'=' * 70}")

        for attribute in attributes:
            print(f"\n{'=' * 60}")
            print(f"Attribute: {attribute}")
            print(f"{'=' * 60}")

            best_f1 = -np.inf
            best_accuracy = -np.inf
            best_recall = -np.inf

            best_model_f1 = None
            best_model_accuracy = None
            best_model_recall = None

            results_by_metric = {
                'F1': {},
                'Accuracy': {},
                'recall_score': {}
            }

            for algo_name, algo_dir in algorithms.items():
                results_file = f'{results_dir}/{algo_dir}/{attribute}_params.json'
                results = json.load(open(results_file, 'r'))

                if 'F1' in results:
                    f1_values = results['F1']
                    f1_idx = np.argmax(f1_values)
                    f1_score = f1_values[f1_idx]
                    f1_params = results['params'][f1_idx]

                    results_by_metric['F1'][algo_name] = {
                            'score': f1_score,
                            'params': f1_params
                    }

                    if f1_score > best_f1:
                        best_f1 = f1_score
                        best_model_f1 = algo_name

                if 'Accuracy' in results:
                    print(attribute)
                    acc_values = results['Accuracy']
                    acc_idx = np.argmax(acc_values)
                    acc_score = acc_values[acc_idx]
                    acc_params = results['params'][acc_idx]

                    results_by_metric['Accuracy'][algo_name] = {
                        'score': acc_score,
                        'params': acc_params
                    }

                    if acc_score > best_accuracy:
                        best_accuracy = acc_score
                        best_model_accuracy = algo_name

                if 'recall_score' in results:
                    recall_values = results['recall_score']
                    recall_idx = np.argmax(recall_values)
                    recall_score = recall_values[recall_idx]
                    recall_params = results['params'][recall_idx]

                    results_by_metric['recall_score'][algo_name] = {
                        'score': recall_score,
                        'params': recall_params
                    }

                    if recall_score > best_recall:
                        best_recall = recall_score
                        best_model_recall = algo_name

            best_models[attribute] = {
                'best_by_F1': {
                    'algorithm': best_model_f1,
                    'F1_score': best_f1,
                    'parameters': results_by_metric['F1'][best_model_f1]['params'] if best_model_f1 else None,
                    'all_scores': {algo: data['score'] for algo, data in results_by_metric['F1'].items()}
                },
                'best_by_Accuracy': {
                    'algorithm': best_model_accuracy,
                    'accuracy_score': best_accuracy,
                    'parameters': results_by_metric['Accuracy'][best_model_accuracy][
                    'params'] if best_model_accuracy else None,
                    'all_scores': {algo: data['score'] for algo, data in results_by_metric['Accuracy'].items()}
                },
                'best_by_Recall': {
                    'algorithm': best_model_recall,
                    'recall_score': best_recall,
                    'parameters': results_by_metric['recall_score'][best_model_recall][
                    'params'] if best_model_recall else None,
                    'all_scores': {algo: data['score'] for algo, data in results_by_metric['recall_score'].items()}
                }
            }


        print(f"\nBest by F1:       {best_model_f1} ({best_f1:.4f})")
        if best_model_f1:
            print(f"  Parameters: {results_by_metric['F1'][best_model_f1]['params']}")

        print(f"\nBest by Accuracy: {best_model_accuracy} ({best_accuracy:.4f})")
        if best_model_accuracy:
            print(f"  Parameters: {results_by_metric['Accuracy'][best_model_accuracy]['params']}")

        print(f"\nBest by Recall:   {best_model_recall} ({best_recall:.4f})")
        if best_model_recall:
            print(f"  Parameters: {results_by_metric['recall_score'][best_model_recall]['params']}")

        json.dump(best_models, open(output_file, 'w'), indent=4)

        print(f"\n{'=' * 70}")
        print(f"Best models and parameters saved to: {output_file}")
        print(f"{'=' * 70}")

        return best_models

    def export_comparison_report(self,attributes: List[str],results_dir: str = './results',output_file: str = 'model_comparison_report.csv'):

        df_results = self.find_best_model_performances(attributes, results_dir)

        df_results.to_csv(output_file, index=False)
        print(f"\nFull comparison report saved to: {output_file}")

        summary_cols = ['Attribute']
        for algo in ['CatBoost', 'HistGradientBoosting', 'RandomForest']:
            for metric in ['Accuracy', 'F1', 'recall_score']:
                summary_cols.append(f'{algo}_{metric}')

        if 'CatBoost_logloss' in df_results.columns:
            summary_cols.append('CatBoost_logloss')

        df_summary = df_results[[col for col in summary_cols if col in df_results.columns]]
        summary_file = output_file.replace('.csv', '_summary.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"Summary report saved to: {summary_file}")

        return df_results, df_summary

    def plot_performances(self,attributes:List[str]):
        for attribute in attributes:
            for algo in ['catboost','XGboost']:
                results = json.load(open(f'./results/{algo}/{attribute}_params.json'))
                x_axis = [str(val) for val in results['params']]

                y_accuracy = results['Accuracy']
                plt.plot(x_axis,y_accuracy)
                plt.xticks(rotation=45)
                plt.show()


## Training
learning = initial_data(learning_vars,is_train=True)
testing = initial_data(test_vars,is_train=False)

print('Processing training dataset')
learning.preprocessing()

print('Processing testing dataset')
testing.preprocessing()

print('Now computing best models')

#for XGB in learning.XGBoosting_search(['main','student','retiree','unemployed']):
#    print(XGB)
#for grad_model in learning.Gradient_Boosting_search(['main','student','retiree','unemployed']):
#    print(grad_model)

#learning.export_comparison_report(attributes=['main','student','retiree','unemployed'])

#best_overall = learning.export_best_overall_parameters(
#        ['main','student','retiree','unemployed'],
#        output_file='./best_model_winner.json'
#    )
#print(best_overall)
best_overall = None
#testing.plot_performances(['main','student','retiree','unemployed'])
testing.plot_performances(['main'])