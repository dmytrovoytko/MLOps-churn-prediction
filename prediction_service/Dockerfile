FROM python:3.11.9-slim

ARG S3_ENDPOINT_URL
ARG S3_BUCKET

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ARG MONGODB_ADDRESS

RUN apt clean
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get -y install unzip nano
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# RUN aws --endpoint-url=http://localstack:4566 s3 mb s3://$S3_BUCKET
# RUN aws --endpoint-url=http://localstack:4566 s3 cp --recursive model s3://$S3_BUCKET
# RUN aws --endpoint-url=http://localstack:4566 s3 ls s3://$S3_BUCKET

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY requirements.txt requirements.txt

RUN uv pip install --system --no-cache-dir -r requirements.txt

# RUN pip install evidently

COPY app.py . 
COPY preprocess.py . 
COPY predict.py . 
COPY settings.py . 
COPY utils.py . 
COPY ./model ./model 
COPY .env .

# temporary
COPY test-api.py . 

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=5000"]

EXPOSE 5555

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5555", "app:app" ]
