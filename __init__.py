import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report
from catboost import CatBoostClassifier, Pool

test_vars = json.load(open('./catboost_params/test_vars.json',"r"))
learning_vars = json.load(open('./catboost_params/learning_vars.json',"r"))
geography = json.load(open('./catboost_params/geography.json',"r"))

def make_path(begin:str, end:str) -> str:
    return os.path.join(os.path.abspath(begin), end)

class initial_data:
    def __init__(self,dataset:pd.DataFrame,is_train:bool):
        self.is_train = is_train
        self.has_been_called = True
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
            self.y = self.main["target"].astype("string").fillna("__MISSING_TARGET__")
            self.main.drop(columns=["target"], inplace=True)
        else:
            self.UID = self.main["UID"].copy()
        self.main.drop(columns=['UID'],inplace=True)
        self.__string_vars =  list(self.main.dtypes == 'object')
#        print(self.main.loc[:,[not x for x in self.__string_vars]].columns)
        self.main.loc[:,self.__string_vars] = self.main.loc[:,self.__string_vars].fillna('Missing').astype(str)
        self.enc.fit(self.main.select_dtypes('object'))
        print('Merging Dataset')
        self.main = pd.concat([self.main.loc[:,[not x for x in self.__string_vars]],
                          pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),
                                       columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),],
                         axis=1) if self.is_train else pd.concat([self.main.loc[:,[not x for x in self.__string_vars]],
                          pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),
                                       columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),],
                         axis=1)
#        return (y,pd.concat([self.main.loc[:,[not x for x in self.__string_vars]],
#                          pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),
#                                       columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),],
#                         axis=1)) if is_train else pd.concat([self.main.loc[:,[not x for x in self.__string_vars]],
#                          pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),
#                                       columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),],
#                         axis=1)
    @check_if_train
    def search_grid(self):
        grid = []
        for depth in [6, 8]:
            for lr in [0.05, 0.1]:
                for l2 in [3, 7]:
                    grid.append({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2})

        classes = sorted(self.y.astype(str).unique().tolist())
        cv = RepeatedKFold(n_splits=5, n_repeats=True, random_state=66)

        results = []
        for g in grid:
            accs, f1s, lls = [], [], []

            for tr, va in cv.split(X = self.main, y = self.y):
                train_pool = Pool(self.main.iloc[tr], self.y.iloc[tr])
                valid_pool = Pool(self.main.iloc[va], self.y.iloc[va])

                model = CatBoostClassifier(
                    loss_function="Logloss",
                    iterations=500,
                    od_type="Iter",        # early stopping
                    od_wait=50,
                    random_seed=66,
                    thread_count=-1,
                    verbose=False,
                    **g
                )
                model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

                pred = model.predict(valid_pool).astype(str).ravel()
                proba = model.predict_proba(valid_pool)

                y_true = self.y.iloc[va].astype(str).values
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
        print(f"Done {g} | F1={row['cv_f1_macro_mean']:.4f} ACC={row['cv_acc_mean']:.4f} LogLoss={row['cv_logloss_mean']:.4f}")
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
        json.dump(best_params,open('./catboost_params/best_params.json','w'))
        print("BEST PARAMS:", best_params)
        print("BEST CV:", {
            "cv_f1_macro_mean": best["cv_f1_macro_mean"],
            "cv_acc_mean": best["cv_acc_mean"],
            "cv_logloss_mean": best["cv_logloss_mean"],
        })

        return best_params

    @check_if_train
    def get_train_split(self):
        return train_test_split(
        self.main, self.y, test_size=0.2, random_state=66, stratify=self.y
        )


## Prediction
learning = initial_data(learning_vars,is_train=True)
testing = initial_data(test_vars,is_train=False)

learning.preprocessing()
print('Processing testing dataset')
testing.preprocessing()

best_params = learning.search_grid()

X_tr, X_va, y_tr, y_va = learning.train_test_split()
tr_pool = Pool(X_tr, y_tr)
va_pool = Pool(X_va, y_va)

cm_model = CatBoostClassifier(**best_params)
cm_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

pred_va = cm_model.predict(va_pool).astype(str).ravel()
y_va_str = y_va.astype(str).values

print("\nConfusion matrix:")
print(confusion_matrix(y_va_str, pred_va))
print("\nClassification report:")
print(classification_report(y_va_str, pred_va))

# final train + predict
full_pool = Pool(learning.main, learning.y)
test_pool = Pool(testing.main)

final_model = CatBoostClassifier(**best_params)
final_model.fit(full_pool)

pred_test = final_model.predict(test_pool).astype(str).ravel()
out = pd.DataFrame({"UID": testing.UID, "target": pred_test})
out.to_csv("predictions.csv", index=False)
print("\nSaved predictions.csv")