#!/bin/bash
echo
echo '1. COPYING LATEST TRAINED MODEL AND SCRIPTS'
echo
cp -r ../train_model/model .
cp ../train_model/preprocess.py .
cp ../train_model/predict.py .
cp ../train_model/settings.py .
cp ../train_model/utils.py .
cp test.env .env

if [[ -e "test.env" ]]
  then
    # loading script parameters from .env
    set -a            
    source test.env
    set +a
else
    echo "No test.env file with paramaters found. Exiting."
    exit 1
fi

# exit 0

echo
echo '2. BUILDING DOCKER IMAGE...'
echo
docker build -t prediction-service .

sleep 5

echo
echo '3. RUNNING DOCKER IMAGE... API available on port 5555'
echo
docker run -p 5555:5555 prediction-service:latest &
# --name prediction-service 


