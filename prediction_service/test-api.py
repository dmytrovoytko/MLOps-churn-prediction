import pandas as pd
import requests
from settings import DATA_DIR, DATASET_NUM, MODEL_DIR, PORT, TARGET

DEBUG = True # True # False # override global settings
# DATASET_NUM = 2 # override for wrong test

from utils import S3_ENDPOINT_URL, S3_BUCKET # isort:skip

if __name__ == '__main__':
    if DATASET_NUM==1:
        # dataset 1
        data = [
        (50001,1,4.0,'Mobile Phone',3,6.0,'Debit Card','Female',3.0,3,'Laptop & Accessory',2,'Single',9,1,11.0,1.0,1.0,5.0,159.93),
        (50004,1,0.0,'Phone',3,15.0,'Debit Card','Male',2.0,4,'Laptop & Accessory',5,'Single',8,0,23.0,0.0,1.0,3.0,134.07),
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
    else:
        print('!!! Predict test, incorrect dataset number:', DATASET_NUM)
        exit(1)

    url = f'http://localhost:{PORT}/predict'
    print('Testing web service', url)
    for row in df.to_dict('records'):
        try:
            response = requests.post(url, json=row)
            print('\n data:', row)
            print(' response:', response.status_code)
            if response.status_code==200:
                print('   source:', row[TARGET], ' -> prediction:', response.json())
            else:
                print('   error:', response.text)
        except Exception as e:
            print('   error:', e)

    # if DEBUG:
    #     # let's intentionally test error request
    #     row = {'test': 1}
    #     print('\n\nLet\'s intentionally test error request:', row)
    #     response = requests.post(url, json=row)
    #     print('\n data:', row)
    #     print(' response:', response.status_code)
    #     print('    error?', response.text)

    # if S3_ENDPOINT_URL:
    #     # Request update model from S3 bucket
    #     url = f'http://localhost:{PORT}/update'
    #     row = {'update': 0}
    #     print('\n\nRequest update model from S3 bucket')
    #     response = requests.post(url, json=row)
    #     print('\n data:', row)
    #     print(' response:', response.status_code)
    #     print('    error?', response.text)

    if DEBUG:
        # Request service status & model info
        url = f'http://localhost:{PORT}/status'
        row = {'status': 0}
        print('\n\nRequest model info')
        response = requests.post(url, json=row)
        print(' response:', response.status_code)
        try:
            if response.status_code==200:
                print('   response:', response.json())
            else:
                print('   error:', response.text)
        except Exception as e:
            print(' ', e)
