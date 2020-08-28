
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def _splitdata(data, label):
       x_train, x_test, y_train, y_test = train_test_split(
           data, label, stratify=label, test_size=.1, random_state=50)
       return x_train, x_test, y_train, y_test


def DealwithSample(data, label, method="ADA"):
    if method== "ADA":
        ada = ADASYN(random_state=42)
        X_res, y_res = ada.fit_resample(data, label)
        return X_res, y_res 
    elif method == "RandomOverSampler" :
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(data, label)
        print("has oversampled the data {}".format(len(y_res)) )
        return X_res, y_res
    elif method == "SMOTE" :
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(data, label)
        return X_res, y_res


