# import warnings # supress warnings
# warnings.filterwarnings('ignore')

import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import OrdinalEncoder

# Display all columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)




load_dotenv()

from settings import DATA_DIR, DATASET_NUM, MODEL_DIR, REMOVE_OUTLIERS, TARGET # isort:skip
from settings import DEBUG # isort:skip
DEBUG = True # True # False # override global settings

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
print(f'ENV: S3_ENDPOINT_URL {S3_ENDPOINT_URL}, S3_BUCKET {S3_BUCKET}\n')
if S3_ENDPOINT_URL:
    import boto3
    S3 = boto3.client(service_name='s3', endpoint_url=S3_ENDPOINT_URL)

def s3_list_buckets(s3):
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    return buckets

def s3_create_bucket(s3, bucket_name):
    s3.create_bucket(Bucket=bucket_name)

def s3_list_objects(s3, bucket_name, filter=''):
    response = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        objects = []
        if filter:
            for obj in response['Contents']:
                if filter in obj['Key']:
                    objects.append(obj['Key'])
        else:
            for obj in response['Contents']:
                objects.append(obj['Key'])
        return objects
    else:
        return []

def s3_upload_files(s3, bucket_name, local_path, prefix_key='', verbose=DEBUG):
    # command = ["aws", "s3", "cp", local_folder, s3_bucket]
    #  try:
    #    subprocess.run(command, check=True)
    #    print("Files copied to S3 successfully!")
    #  except subprocess.CalledProcessError as e:
    #    print("Error:", e)
    if os.path.isdir(local_path):
        # prefix_key = '' # ?
        if local_path[-1]=='/':
            local_path = local_path[:-1]
        for file_name in os.listdir(local_path):
            if verbose:
                print("Uploading:", local_path+'/'+file_name)
            if os.path.isdir(local_path+'/'+file_name):
                if verbose:
                    print("Uploading subdir:", local_path+'/'+file_name)
                s3_upload_files(s3, bucket_name, local_path+'/'+file_name, prefix_key=prefix_key+file_name+'/', verbose=DEBUG)
            else:
                s3.upload_file(local_path+'/'+file_name, bucket_name, prefix_key + file_name)
    else:
        file_name = os.path.basename(local_path) # local_path.split('/')[-1]
        if verbose:
            print("Uploading:", local_path)
        s3.upload_file(local_path, bucket_name, prefix_key + file_name)

def s3_download_file(s3, bucket_name, object_name, file_name, verbose=DEBUG):
    if verbose:
        print("Downloading:", object_name)
    s3.download_file(bucket_name, object_name, file_name)


def enc_save(enc, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)

def enc_load(file_name):
    # if S3_ENDPOINT_URL:
    #     try:
    #         s3_download_file(S3, S3_BUCKET, 'model/encoder.pkl', f'{MODEL_DIR}encoder.pkl', verbose=DEBUG)
    #     except Exception as e:
    #         print(f'! Error S3_ENDPOINT_URL {S3_ENDPOINT_URL}, S3_BUCKET {S3_BUCKET}\n', e)

    with open(file_name, 'rb') as f:
        enc = pickle.load(f)
        return enc
    return OrdinalEncoder()



if __name__ == '__main__':

    if not S3_ENDPOINT_URL:
        exit()

    file_name = f'model.pkl'
    try:
        # import s3fs
        # s3 = s3fs.S3FileSystem() #anon=True)
        # print(s3.ls(S3_BUCKET))
        # with open(f"s3://{S3_BUCKET}/{file_name}", 'rb') as data:
        #     print(len(data))

        s3 = boto3.client(service_name='s3', endpoint_url=S3_ENDPOINT_URL)
        buckets = s3_list_buckets(s3)
        print('buckets:', buckets)

        # create bucket
        bucket_name = S3_BUCKET
        if not bucket_name in buckets:
            s3.create_bucket(Bucket=bucket_name)
            buckets = s3_list_buckets(s3)
            print(S3_BUCKET, 'created?', buckets)

        # options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
        # url = f"s3://{S3_BUCKET}/{filename}"
        # file_name = 'settings.py'
        # boto3.resource is not well depeloved anymore
        # s3 = boto3.resource('s3', endpoint_url=S3_ENDPOINT_URL)
        # obj = s3.Object(bucket_name=S3_BUCKET, key=file_name)
        # print(obj.bucket_name)
        # print(obj.key)
        # myBucket = s3.Bucket(S3_BUCKET)
        # for object_summary in myBucket.objects.filter(Prefix="model/"):
        #     print(object_summary.key)
        # print('--------')

        # key = 'model/'+file_name
        # print(key)
        # obj = myBucket.Object(key).get()
        # enc = pickle.load(obj)
        # print(type(enc))
        # response = obj.get()
        # data = response['Body'].read()
        # print(data)
        # copy file from S3 bucket to a file_name place
        # https://stackoverflow.com/questions/48964181/how-to-load-a-pickle-file-from-s3-to-use-in-aws-lambda
        prefix='model/'
        objects = s3_list_objects(s3, bucket_name, filter='')
        print('objects:', objects)
        print('--------')

        local_path = 'settings.py'
        s3_upload_files(s3, bucket_name, local_path)
        objects = s3_list_objects(s3, bucket_name, filter='')
        print('objects:', objects)
        print('--------')

        local_path = './model/2/model.pkl'
        s3_upload_files(s3, bucket_name, local_path, prefix_key='model/')
        objects = s3_list_objects(s3, bucket_name, filter='')
        print('objects:', objects)
        print('--------')

        objects = s3_list_objects(s3, bucket_name, filter='model/')
        print('objects:', objects)
        print('--------')


        local_path = './model/2/'
        s3_upload_files(s3, bucket_name, local_path, prefix_key='model/')
        objects = s3_list_objects(s3, bucket_name, filter='')
        print('objects:', objects)
        print('--------')

        s3_download_file(s3, bucket_name, 'model/model.pkl', './model/model.pkl')

        # with open('_'+file_name, 'wb') as f:
        #     s3c.download_fileobj(S3_BUCKET, 'model/'+file_name, f)
    except Exception as e:
        print(f'! Error S3_ENDPOINT_URL {S3_ENDPOINT_URL}, S3_BUCKET {S3_BUCKET}\n', e)
