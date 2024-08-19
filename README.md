# MLOps project Training and Deployment of Churn prediction model

Pet project / Capstone project (2nd) for DataTalks.Club MLOps ZoomCamp`24:

![MLOps project Churn prediction](/screenshots/mlops-churn-prediction.png)

Several models trained and optimized on 2 Churn datasets
1. [Ecommerce Customer Churn Analysis and Prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data)
2. [DQLab Telco Final](https://www.kaggle.com/datasets/samran98/customer-churn-telco-final/data)

3 classifiers used for prediction:
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
3. Run `pipenv install --dev` to install required packages.
4. If you want to play with/develop the project, you can also install `pipenv run pre-commit install` to format code before committing to repo.

#### Virtual machine (AWS, Azure, GCP)

- Create virtual machine
- Clone repositoty
- steps 3. and 4. from above

#### Local machine (Linux)

- Clone repositoty
- steps 3. and 4. from above

### :arrow_heading_down: Dataset

Dataset files are small enough, included in repo. Located in `./train_model/data/` directory.
