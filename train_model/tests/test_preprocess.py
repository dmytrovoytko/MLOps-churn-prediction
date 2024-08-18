import pandas as pd

from preprocess import load_dataset, preprocess_data, enc_load, model_dir
from settings import DATASET_NUM, DATA_DIR, TARGET


def test_load_dataset():
    # fmt: off
    if DATASET_NUM==1:
        # dataset 1
        expected_columns = [
        'CustomerID','Churn','Tenure','PreferredLoginDevice','CityTier','WarehouseToHome','PreferredPaymentMode','Gender',
        'HourSpendOnApp','NumberOfDeviceRegistered','PreferedOrderCat','SatisfactionScore','MaritalStatus',
        'NumberOfAddress','Complain','OrderAmountHikeFromlastYear','CouponUsed','OrderCount',
        'DaySinceLastOrder','CashbackAmount'
        ]
        df = load_dataset(1)
    elif DATASET_NUM==2:
        # dataset 2
        expected_columns = [
        'UpdatedAt','customerID','gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV',
        'InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn',
        ]
        df = load_dataset(2)
    else:
        print('Incorrect dataset number:', DATASET_NUM)
        assert False
    # fmt: on

    assert df.shape[0] > 0
    actual_columns = list(df.columns)
    assert TARGET in actual_columns
    assert len(actual_columns) == len(expected_columns)
    assert all([a == b for a, b in zip(actual_columns, expected_columns)])

    print(all([a == b for a, b in zip(actual_columns, expected_columns)]))


def test_preprocess_data():
    # fmt: off
    if DATASET_NUM==1:
        # dataset 1
        data = [
        (50001,1,4.0,'Mobile Phone',3,6.0,'Debit Card','Female',3.0,3,'Laptop & Accessory',2,'Single',9,1,11.0,1.0,1.0,5.0,159.93),
        (50002,1,None,'Phone',1,8.0,'UPI','Male',3.0,4,'Mobile',3,'Single',7,1,15.0,0.0,1.0,0.0,120.9),
        (50030,0,5.0,'Computer',3,14.0,'E wallet','Female',2.0,3,'Fashion',2,'Single',2,0,14.0,2.0,3.0,7.0,189.98),
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
    df = preprocess_data(df, ord_enc, DATASET_NUM, fit_enc=False)
    assert df.shape[0] > 0
    actual_columns = list(df.columns)
    assert TARGET in actual_columns
    assert len(actual_columns) == len(expected_columns)
    assert all([a == b for a, b in zip(actual_columns, expected_columns)])

    print(all([a == b for a, b in zip(actual_columns, expected_columns)]))
