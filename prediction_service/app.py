import json
import os

import pandas as pd
import requests
from flask import Flask, jsonify, redirect, request, url_for

from settings import DATASET_NUM, DATA_DIR, MODEL_DIR, MODEL_PREFIX, TARGET, PORT # isort:skip
from settings import NEW_API_PARAM_NUM # isort:skip

from settings import DEBUG # isort:skip
DEBUG = True # True # False # override global settings

from utils import S3_ENDPOINT_URL, S3_BUCKET # isort:skip

from predict import predict_dict
from preprocess import enc_load, preprocess_data

app = Flask('online-prediction')

def get_model_info(model_dir=MODEL_DIR):
    try:
        model_info = {}
        with open(f'{MODEL_DIR}MLmodel', 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        for param in ['mlflow_version', 'model_size_bytes', 'model_uuid', 'run_id', 'utc_time_created']:
            line = [line for line in lines if line.startswith(param)]
            if line:
                value = line[0].split(': ')[-1].strip("''")
                model_info.update({param: value})
        if DEBUG:
            print((model_info))
    except Exception as e:
        print(e)
    return model_info


@app.route('/status', methods=['POST'])
def service_status():
    object = request.get_json()
    if DEBUG:
        print('STATUS Received request:', object)
    model_info = get_model_info()
    model_info.update({ "DATASET_NUM": int(DATASET_NUM),
                        "MODEL_DIR": MODEL_DIR,
                        "S3_ENDPOINT_URL": S3_ENDPOINT_URL,
                        "S3_BUCKET": S3_BUCKET,
                    })
    return jsonify(model_info)


@app.route('/update', methods=['POST'])
def update_model():
    object = request.get_json()
    if DEBUG:
        print('UPDATE_MODEL Received request:', object)

    if S3_ENDPOINT_URL:
        from utils import S3 as s3
        from utils import s3_download_file, s3_list_buckets, s3_list_objects
        buckets = s3_list_buckets(s3)
        bucket_name = S3_BUCKET
        if not bucket_name in buckets:
            return f"S3 bucket ({S3_BUCKET}) not found, update failed", 400 #status.HTTP_400_BAD_REQUEST
        if DEBUG:
            objects = s3_list_objects(s3, bucket_name, filter='model/')
            print('\n--------')
            print('S3 objects:', sorted(objects))
            print('--------')
        try:
            s3_download_file(s3, bucket_name, f'{MODEL_PREFIX}encoder.pkl', f'{MODEL_DIR}encoder.pkl', verbose=DEBUG)
            s3_download_file(s3, bucket_name, f'{MODEL_PREFIX}model.pkl', f'{MODEL_DIR}model.pkl', verbose=DEBUG)
            s3_download_file(s3, bucket_name, f'{MODEL_PREFIX}MLmodel', f'{MODEL_DIR}MLmodel', verbose=DEBUG)
            result = { 'updated': int(DATASET_NUM) }
        except Exception as e:
            return "Downloading update from S3 failed: "+str(e), 500 #status.HTTP_400_BAD_REQUEST
    else:
        return "Empty S3 settings, update failed", 400 #status.HTTP_400_BAD_REQUEST
    return jsonify(result)

def calculate_prediction(object, dataset_num):
    try:
        result = predict_dict(object, dataset_num, verbose=True)
    except Exception as e:
        try:
            new_dataset_num = (3-dataset_num) # we have 2 datasets, so it would be 1 for 2, or 2 for 1
            print(f'! Error: {e}\nTrying to use the other model: {new_dataset_num} instead of {dataset_num}')
            result = predict_dict(object, new_dataset_num, verbose=True)
        except Exception as e:
            print(f'!! Error: {e}')
            return "Wrong data/dataset not recognized", 400 #status.HTTP_400_BAD_REQUEST

    # TODO check params -> store request data to DB for monitoring
    return jsonify(result)


@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    object = request.get_json()
    if DEBUG:
        print('Received request v1:', object)
    return calculate_prediction(object, 1)

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    object = request.get_json()
    if DEBUG:
        print('Received request v2:', object)
    return calculate_prediction(object, 2)

@app.route('/predict', methods=['POST'])
def predict():
    object = request.get_json()
    if DEBUG:
        print('Received request:', object)

    # detect request data change and switch API
    if len(object)<=NEW_API_PARAM_NUM:
        dataset_num = 2
    else:
        dataset_num = 1
    url = url_for(f'predict_v{dataset_num}') #, variable=object)
    if DEBUG:
        print('Using API', dataset_num, url)
    return redirect(url, code=307) # 302 redirects to GET, 307 preserves original

    return calculate_prediction(dataset_num)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=PORT)
