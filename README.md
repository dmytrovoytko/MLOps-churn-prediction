# MLOps project Training and Deployment of Churn prediction model

Pet project / Capstone project (2nd) for DataTalks.Club MLOps ZoomCamp`24:

![MLOps project Churn prediction](/screenshots/mlops-churn-prediction.png)

Several models trained and optimized on 2 Churn datasets
1. [Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data)
2. [DQLab Telco Final](https://www.kaggle.com/datasets/samran98/customer-churn-telco-final/data)

3 classifiers used for prediction (known as effective for churn prediction):
- DecisionTreeClassifier
- RandomForestClassifier
- XGBClassifier

Project can be tested and deployed in cloud virtual machine (AWS, Azure, GCP), **GitHub CodeSpaces** (the easiest option, and free), or locally (GPU is not required).

To reproduce and review this project it would be enough less than an hour. For GitHub CodeSpace option you don't need to use anything extra at all - just your favorite web browser + GitHub account is totally enough.

## Problem statement

Harvard Business Review: "Depending on which study you believe, and what industry you’re in, acquiring a new customer is anywhere from 5 to 25 times more expensive than retaining an existing one. It makes sense: you don’t have to spend time and resources going out and finding a new client — you just have to keep the one you have happy". Other statistics show an increase in customer retention by 5% can lead to a company’s profits growing by 25% to around 95% over a period of time. Churn rate is an efficient indicator for subscription-based companies. So I decided to use Machine Learning to predict customer churn using data collected from e-commerce.

## 🎯 Goals

This is my 2nd MLOps project started during [MLOps ZoomCamp](https://github.com/DataTalksClub/mlops-zoomcamp)'24.

And **the main goal** is straight-forward: build an end-to-end Machine Learning project:
- choose dataset
- load & analyze data, preprocess it
- train & test ML model
- create a model training pipeline
- deploy the model (as a web service)
- finally monitor performance
- And follow MLOps best practices!

I found a Churn dataset [Ecommerce Customer Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data) on Kaggle, analyzed and experimented with different classification models. And I was surprised how hyper parameter optimization could improve results!
I was curious enough to find another churn dataset to experiment with it as well. It has a different set of features (only few are similar, like tenure, gender). As a result I managed to build more or less universal pipeline for this task that can easily switch datasets, and I learned a lot while making it possible.

Thanks to MLOps ZoomCamp for the reason to learn many new tools!

## :toolbox: Tech stack

- MLFlow for ML experiment tracking
- Prefect for ML workflow orchestration
- Docker and docker-compose
- Localstack as AWS S3 service emulation for development
- Flask for web application
- MongoDB + WhyLogs for performance monitoring

## 🚀 Instructions to reproduce

- [Setup environment](#hammer_and_wrench-setup-environment)
- [Dataset](#arrow_heading_down-dataset)
- [Train model](#train-model)
- [Test prediction service](#test-prediction-service)
- [Deployment and Monitoring](#deployment-and-monitoring)
- [Best practices](#best-practices)

### :hammer_and_wrench: Setup environment

#### Github CodeSpaces

1. Fork this repo on GitHub.
2. Create GitHub CodeSpace from the repo, **Start CodeSpace**
3. Run `pip install -r requirements.txt` to install required packages.
4. If you want to play with/develop the project, you can also install `pipenv run pre-commit install` to format code before committing to repo.

#### Virtual machine (AWS, Azure, GCP)

- Create virtual machine
- Clone repositoty, `cd MLOps-churn`
- steps 3. and 4. from above

#### Local machine (Linux)

- Clone repositoty, `cd MLOps-churn`
- steps 3. and 4. from above

NB Tested with python 3.11 and 3.12. As packages versions might conflict with yours, Github CodeSpaces could be a good solution. Or you can just create a new virtual environment using `python -m venv mlopsenv` and then `source mlopsenv/bin/activate`, or by using `pipenv install`.

### :arrow_heading_down: Dataset

Dataset files are small enough, included in repo. Located in `./train_model/data/` directory.
You can switch dataset in `train_model/settings.py`, as well as paths and other parameters.
Data preprocessing includes removing some unuseful columns, fixing missing values and encoding categorical columns. Encoders are different for each dataset as columns set differs, so stored in separate directories (with models).

![Dataset details](/screenshots/dataset-info-1.png)

### Train model

Run `bash run-train-model.sh` or go to `train_model` directory and run `python orchestrate.py`.
This will start Prefect workflow to
- load training data (dataset 1 or 2 according to `settings.py`), training encoder
- call `run_experiment()` with different hyper parameters and showing accuracy results (dataset is split - train, test)
- call `run_register_model()` to register the best model, which will be saved to `./model` sub-directory (1 or 2)
- call `test_model()` to verify accuracy on the whole dataset

To explore results go to `train_model` directory and run `mlflow server`.

NB Your local install/environment might require starting prefect server before running `bash run-train-model.sh`, you can do it by `prefect server start`.

![MLFlow experiments training churn model](/screenshots/mlflow-experiments-2.png)


#### Prefect orchestration

Prefect workflow is located in `orchestrate.py`.
It trains 3 classifiers - DecisionTree, RandomForest and XGBoost. DecisionTree is very fast end quite effective, the others require more time, so I disabled them for your convenience:
```
    for classifier in [
        'DecisionTreeClassifier',
        # 'RandomForestClassifier',
        # 'XGBClassifier',
    ]:
```
Feel free to uncomment and test in full.

![Prefect orchestration](/screenshots/prefect-runs.png)

Full training includes 9 experiments - 3 classifiers x 3 estimators, and ability to set ranges for other hyper parameters.
Inside the experiment (`train_model()` in `train_model.py`) are more specific hyper parameners for the classifiers.

### Test prediction service

Integration test is very similar to deployment.
Test parameters are set in `test.env` file, including settings for AWS account and S3 bucket. You need to set them correctly for your deployment, otherwise it work with localstack emulation of AWS S3 service.

Run `bash test-service.sh` or go to `prediction_service` directory and run `bash test-run.sh`.
This will
- copy best model and latest scripts,
- build docker image,
- run it, and
- run `test-api.py` to execute requests to the web service.
Finally docker container will be stopped.

![Testing prediction service in dockerl](/screenshots/prediction-service-test-1.png)

![Testing prediction service in dockerl](/screenshots/prediction-service-test-dataset-1.png)

Advanced testing can be executed by running `docker compose up --build` in `prediction_service` (check `docker-compose.yaml` settings - MongoDB, Localstack).

### Deployment and Monitoring

To deploy web service set your parameters in `test.env` file, then run `bash deploy-service.sh`.

Monitoring is made by storing requests and predictions in MongoDb database, then using WhyLogs (`data-drift-test.py`) to check data/prediction drift [example](/screenshots/example2.html).

### Best practices

    * [x] Unit tests
    * [x] Integration test (== Test prediction service)
    * [x] Code formatter (isort, black)
    * [x] Makefile
    * [x] Pre-commit hooks
    * [x] Github workflow for testing on push/pull request


## Current results of training

By using 3 classifiers and tuning different hyper parameters I managed to achive 99% accuracy for dataset 1.
The best results achieved by using XGBClassifier.
To be honest I was surprised how those hyperparameters affect prediction accuracy! You have very low chances to find optimal combination just by playing with Jupyter notebooks - MLFlow rules!
Another surprise is that DecisionTreeClassifier can be quite close in accuracy with much faster execution! Of course, it depends on dataset.

![Trained churn model: results](/screenshots/prediction-accuracy-dataset1.png)

You can find additional information which parameners result better performance on [screenshots](/screenshots).

As I mentioned, I experimented with 2 datasets, and made web service flexible enough to

- recognize change of dataset and redirect to respective model prediction
- update model files from S3 bucket, making possible to monitor data, retrain model and command service to 'upgrade' without restarting the service.
- check service parameters by /status request (check `app.py`)

That was fun!

## Next steps

What's interesting about churn prediction? I found another dataset - new experiments ahead!

So stay tuned! (you can ⭐️star the repo to be notified about updates).

## Support

🙏 Thank you for your attention and time!

- If you experience any issue while following this instruction (or something left unclear), please add it to [Issues](/issues), I'll be glad to help/fix. And your feedback, questions & suggestions are welcome as well!
- Feel free to fork and submit pull requests.

If you find this project helpful, please ⭐️star⭐️ my repo
https://github.com/dmytrovoytko/MLOps-churn-prediction to help other people discover it 🙏

Made with ❤️ in Ukraine 🇺🇦 Dmytro Voytko
