import pickle

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocess import enc_load, preprocess_data
from settings import DATA_DIR, DATASET_NUM, MODEL_DIR, TARGET

from settings import DEBUG # isort:skip
# DEBUG = True # True # False # override global settings

def print_results(classifier, y_test, predict_y, verbose=DEBUG):
    print(f'\n================\n{classifier}')
    if verbose:
        print(classification_report(y_test, predict_y, digits=3))
        print(confusion_matrix(y_test, predict_y))
    print(f' accuracy: {accuracy_score(y_test, predict_y):05.3f}')
    print(f' balanced accuracy: {balanced_accuracy_score(y_test, predict_y):05.3f}')

def predict_df(df, MODEL_DIR, verbose=DEBUG):
    print(f'\nPredicting using model {MODEL_DIR}')

    try:
        test_data = preprocess_data(df)
    except:
        test_data = df

    cols = test_data.columns.to_list()
    X_test = pd.DataFrame(test_data, columns=cols)
    if TARGET in cols:
        y_test = X_test.pop(TARGET)
    else:
        y_test = pd.Series()

    try:
        model = pickle.load(open(f'{MODEL_DIR}model.pkl', 'rb'))
        # print('model.get_params:', model.get_params())
    except Exception as e:
        print('!!! Exception while loading model:', e)
        return pd.Series()

    try:
        y_pred = model.predict(X_test)
        if verbose:
            estimator = type(model.get_params().get('estimator', ''))
            print(f"\nPrediction:\n{estimator}\ny_pred: {list(y_pred)[:10]}")
            if len(y_test):
                print(f"y_test: {list(y_test)[:10]}")
        return y_pred
    except Exception as e:
        print(f'!!! Exception while predicting {TARGET}:', e)
        return pd.Series()



if __name__ == '__main__':
    if DATASET_NUM==1:
        # dataset 1
        data = [
        (50001,1,4.0,'Mobile Phone',3,6.0,'Debit Card','Female',3.0,3,'Laptop & Accessory',2,'Single',9,1,11.0,1.0,1.0,5.0,159.93),
        (50002,1,None,'Phone',1,8.0,'UPI','Male',3.0,4,'Mobile',3,'Single',7,1,15.0,0.0,1.0,0.0,120.9),
        (50003,1,None,'Phone',1,30.0,'Debit Card','Male',2.0,4,'Mobile',3,'Single',6,1,14.0,0.0,1.0,3.0,120.28),
        (50004,1,0.0,'Phone',3,15.0,'Debit Card','Male',2.0,4,'Laptop & Accessory',5,'Single',8,0,23.0,0.0,1.0,3.0,134.07),
        (50005,1,0.0,'Phone',1,12.0,'CC','Male',None,3,'Mobile',5,'Single',3,0,11.0,1.0,1.0,3.0,129.6),
        (50006,1,0.0,'Computer',1,22.0,'Debit Card','Female',3.0,5,'Mobile Phone',5,'Single',2,1,22.0,4.0,6.0,7.0,139.19),
        (50030,0,5.0,'Computer',3,14.0,'E wallet','Female',2.0,3,'Fashion',2,'Single',2,0,14.0,2.0,3.0,7.0,189.98),
        (50031,0,2.0,'Computer',1,6.0,'COD','Male',2.0,3,'Laptop & Accessory',3,'Divorced',2,0,13.0,0.0,1.0,9.0,143.19),
        ]

        columns = [
        'CustomerID','Churn','Tenure','PreferredLoginDevice','CityTier','WarehouseToHome','PreferredPaymentMode','Gender',
        'HourSpendOnApp','NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore','MaritalStatus',
        'NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
        'DaySinceLastOrder','CashbackAmount'
        ]
        df = pd.DataFrame(data, columns=columns)
        # df.drop(['CustomerID', 'Churn'], axis=1, inplace = True)
        dataset_num = 1
        ord_enc = enc_load(f'{DATA_DIR}encoder{dataset_num}.pkl')
        df = preprocess_data(df, ord_enc, dataset_num, fit_enc=False)
        # print('OrdinalEncoder categories:', ord_enc.categories_)
        # # df = preprocess_data(df, ord_enc, 1, fit_enc=True)
        # import numpy as np
        # categorical_features = df.select_dtypes(exclude=[np.number]).columns
        # print(categorical_features)
        # df[categorical_features] = ord_enc.fit_transform(df[categorical_features])

        print(df.to_string())
        # exit()
    elif DATASET_NUM==2:
        # dataset 2
        data = [
        (202006,45759018157,'Female','No','Yes',1,'No','No','Yes','Yes',29.85,29.85,'No'),
        (202006,45315483266,'Male','No','Yes',60,'Yes','No','No','Yes',20.5,1198.8,'No'),
        (202006,45236961615,'Male','No','No',5,'Yes','Yes','Yes','No',104.1,541.9,'Yes'),
        (202006,45929827382,'Female','No','Yes',72,'Yes','Yes','Yes','Yes',115.5,8312.75,'No'),
        (202006,45305082233,'Female','No','Yes',56,'Yes','Yes','Yes','No',81.25,4620.4,'No'),
        ]

        columns = [
        'UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV',
        'InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn',
        ]
        df = pd.DataFrame(data, columns=columns)
        # df.drop(['CustomerID', 'Churn'], axis=1, inplace = True)
        dataset_num = 2
        ord_enc = enc_load(f'{DATA_DIR}encoder{dataset_num}.pkl')
        df = preprocess_data(df, ord_enc, dataset_num, fit_enc=False)
        # print('OrdinalEncoder categories:', ord_enc.categories_)
        # # df = preprocess_data(df, ord_enc, 1, fit_enc=True)
        # import numpy as np
        # categorical_features = df.select_dtypes(exclude=[np.number]).columns
        # print(categorical_features)
        # df[categorical_features] = ord_enc.fit_transform(df[categorical_features])

        print(df.to_string())
        # exit()
    else:
        print('!!! Predict test, incorrect dataset number:', DATASET_NUM)
        exit(1)

    y_pred = predict_df(df, MODEL_DIR, verbose=True)

    columns = df.columns.to_list()
    if TARGET in columns:
        print_results('Saved model', df[TARGET], y_pred, verbose=True)
