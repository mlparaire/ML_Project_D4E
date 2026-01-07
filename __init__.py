import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, classification_report

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

    def preprocessing(self) -> pd.DataFrame:
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.__string_vars =  list(self.main.dtypes == 'object')
        self.main.loc[:,self.__string_vars] = self.main.loc[:,self.__string_vars].fillna('Missing').astype(str)
        self.enc.fit(self.main.select_dtypes('object'))

        print(self.main.loc[:,[not x for x in self.__string_vars]].merge(pd.DataFrame(self.enc.transform(self.main.select_dtypes('object')).toarray(),columns = self.enc.get_feature_names_out(self.main.select_dtypes('object').columns)),
                                                                         right_index=True))

        #print(pd.DataFrame(self.transform.toarray()))
#        self.transformed = pd.DataFrame(self.enc.fit_transform(self.main.loc[:,self.__string_vars]))
#        print(self.transformed)



#        self.__cat_indx = [self.main.columns.get_loc(x) for x in self.__string_vars]
#        print(type(self.__string_vars))
learning = initial_data(learning_vars)
learning.preprocessing()
7