import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from catboost import CatBoostClassifier, Pool

learning_vars = {
        'main' : "learn_dataset.csv",
        'job' : "learn_dataset_job.csv",
        'job_security' : "learn_dataset_JOB_SECURITY.csv",
        'retired_jobs' : "learn_dataset_retired_jobs.csv",
        'retired_formers' : "learn_dataset_retired_former.csv",
        'retired_pension' : "learn_dataset_JOB_SECURITY.csv",
        'sport': "learn_dataset_sport.csv",
}

test_vars = {
    'main': "test_dataset.csv",
    'job': "test_dataset_job.csv",
    'job_security': "test_dataset_JOB_SECURITY.csv",
    'retired_jobs': "test_dataset_retired_jobs.csv",
    'retired_formers': "test_dataset_retired_former.csv",
    'retired_pension': "test_dataset_JOB_SECURITY.csv",
    'sport': "test_dataset_sport.csv",
}

geography = {
    'adm' : 'city_adm.csv',
    'loc' : 'city_loc.csv',
    'pop' : 'city_pop.csv'
}




def make_path(begin:str, end:str) -> str:
    return os.path.join(os.path.abspath(begin), end)

class initial_data:
    def __init__(self,dataset):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
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

    def preprocessing(self,is_train: bool):
        self.enc = OneHotEncoder(handle_unknown='ignore')
        y = self.main["target"].astype("string").fillna("__MISSING_TARGET__")
        self.main.drop(columns=["target"],inplace=True)
        self.__string_vars =  list(self.main.dtypes == 'object')
#        print(self.main.loc[:,[not x for x in self.__string_vars]].columns)
        self.main.loc[:,self.__string_vars] = self.main.loc[:,self.__string_vars].fillna('Missing').astype(str)
        self.enc.fit(self.main.select_dtypes('object'))
        print('Merging Dataset')
        return (self.main.loc[:,self.__string_vars].columns,self.__string_vars,y,pd.concat([self.main.loc[:,[not x for x in self.__string_vars]],
                          pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),
                                       columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),],
                         axis=1)) if is_train else pd.concat([self.main.loc[:,[not x for x in self.__string_vars]],
                          pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),
                                       columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),],
                         axis=1)

    def grid_search_catboost(X, y, cat_idx, seed=66):
        # 8 combinations
        grid = []
        for depth in [6, 8]:
            for lr in [0.05, 0.1]:
                for l2 in [3, 7]:
                    grid.append({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2})

        classes = sorted(y.astype(str).unique().tolist())
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        results = []
        for g in grid:
            accs, f1s, lls = [], [], []

            for tr, va in cv.split(X, y):
                train_pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=cat_idx)
                valid_pool = Pool(X.iloc[va], y.iloc[va], cat_features=cat_idx)

                model = CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=500,
                    od_type="Iter",  # early stopping
                    od_wait=50,
                    random_seed=seed,
                    thread_count=-1,
                    verbose=False,
                    **g
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

                pred = model.predict(valid_pool).astype(str).ravel()
                proba = model.predict_proba(valid_pool)

                y_true = y.iloc[va].astype(str).values
                accs.append(accuracy_score(y_true, pred))
                f1s.append(f1_score(y_true, pred, average="macro"))
                lls.append(log_loss(y_true, proba, labels=classes))

            row = {
                **g,
                "cv_acc_mean": float(np.mean(accs)),
                "cv_f1_macro_mean": float(np.mean(f1s)),
                "cv_logloss_mean": float(np.mean(lls)),
            }
            results.append(row)
            print(
                f"Done {g} | F1={row['cv_f1_macro_mean']:.4f} ACC={row['cv_acc_mean']:.4f} LogLoss={row['cv_logloss_mean']:.4f}")

        res_df = pd.DataFrame(results).sort_values(
            by=["cv_f1_macro_mean", "cv_logloss_mean"],
            ascending=[False, True]
        ).reset_index(drop=True)

        res_df.to_csv("grid_results.csv", index=False)
        print("Saved grid_results.csv")

        best = res_df.iloc[0].to_dict()
        best_params = {
            "depth": int(best["depth"]),
            "learning_rate": float(best["learning_rate"]),
            "l2_leaf_reg": float(best["l2_leaf_reg"]),
        }
        print("BEST PARAMS:", best_params)
        print("BEST CV:", {
            "cv_f1_macro_mean": best["cv_f1_macro_mean"],
            "cv_acc_mean": best["cv_acc_mean"],
            "cv_logloss_mean": best["cv_logloss_mean"],
        })
        return best_params, res_df

learning = initial_data(learning_vars)
testing = initial_data(test_vars)

y,train = learning.preprocessing(is_train=True)
test = learning.preprocessing(is_train=False)

test_uid = test["UID"].copy()

train.drop(columns=["UID"], inplace=True)
test.drop(columns=["UID"], inplace=True)

best_params, _ = grid_search_catboost(X, y, cat_idx, seed=SEED)

# other fixed training params
base_params = dict(
    loss_function="Logloss",
    iterations=500,
    od_type="Iter",
    od_wait=50,
    random_seed=SEED,
    thread_count=-1,
    verbose=False,
    **best_params
)
#print(train.columns)


