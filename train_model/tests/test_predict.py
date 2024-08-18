import pandas as pd

from predict import predict_df
from preprocess import enc_load, preprocess_data, model_dir
from settings import DATASET_NUM, DATA_DIR, MODEL_DIR, TARGET


def test_predict_df():
    # fmt: off
    if DATASET_NUM==1:
        # dataset 1
        data = [
        (50001,1,4.0,'Mobile Phone',3,6.0,'Debit Card','Female',3.0,3,'Laptop & Accessory',2,'Single',9,1,11.0,1.0,1.0,5.0,159.93),
        (50002,1,None,'Phone',1,8.0,'UPI','Male',3.0,4,'Mobile',3,'Single',7,1,15.0,0.0,1.0,0.0,120.9),
        (50003,1,None,'Phone',1,30.0,'Debit Card','Male',2.0,4,'Mobile',3,'Single',6,1,14.0,0.0,1.0,3.0,120.28),
        (50030,0,5.0,'Computer',3,14.0,'E wallet','Female',2.0,3,'Fashion',2,'Single',2,0,14.0,2.0,3.0,7.0,189.98),
        (50031,0,2.0,'Computer',1,6.0,'COD','Male',2.0,3,'Laptop & Accessory',3,'Divorced',2,0,13.0,0.0,1.0,9.0,143.19),
        ]
        columns = [
        'CustomerID','Churn','Tenure','PreferredLoginDevice','CityTier','WarehouseToHome','PreferredPaymentMode','Gender',
        'HourSpendOnApp','NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore','MaritalStatus',
        'NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
        'DaySinceLastOrder','CashbackAmount'
        ]
        expected_columns = [
        # 'CustomerID', - must be dropped
        'Churn','Tenure','PreferredLoginDevice','CityTier','WarehouseToHome','PreferredPaymentMode','Gender',
        'HourSpendOnApp','NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore','MaritalStatus',
        'NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
        'DaySinceLastOrder','CashbackAmount'
        ]
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
        expected_columns = [
        # 'UpdatedAt','customerID', - must be dropped
        'gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV',
        'InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn',
        ]
    else:
        print('Incorrect dataset number:', DATASET_NUM)
        assert False
    # fmt: on

    df = pd.DataFrame(data, columns=columns)
    ord_enc = enc_load(f'{model_dir(DATASET_NUM)}encoder.pkl')

    # test 1: with target labels - as while training models
    df = preprocess_data(df, ord_enc, DATASET_NUM, fit_enc=False)
    assert df.shape[0] > 0
    actual_columns = list(df.columns)
    assert len(actual_columns) == len(expected_columns)
    assert all([a == b for a, b in zip(actual_columns, expected_columns)])

    print(all([a == b for a, b in zip(actual_columns, expected_columns)]))

    actual_result = predict_df(df, MODEL_DIR, verbose=True)
    assert len(actual_result) == len(data)
    for label in list(actual_result):
        assert label in [0, 1]

    # test 2: without target labels - as while real prediction
    # columns = [col for col in columns if col != TARGET]
    df = pd.DataFrame(data, columns=columns)
    df = df.drop([TARGET], axis=1)
    df = preprocess_data(df, ord_enc, DATASET_NUM, fit_enc=False)
    assert df.shape[0] > 0
    actual_result = predict_df(df, MODEL_DIR, verbose=True)
    for label in list(actual_result):
        assert label in [0, 1]
