LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
# LOCAL_IMAGE_NAME:=prediction-service:${LOCAL_TAG}
LOCAL_IMAGE_NAME:=prediction-service:latest

test:
	cd train_model && pipenv run pytest tests/

quality_checks:
	pipenv run isort train_model
	pipenv run black train_model
# 	pipenv run pylint --recursive=y train_model/tests/

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash test-service.sh

publish: integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash deploy.sh

setup:
	pipenv install --dev
	pipenv run pre-commit install
