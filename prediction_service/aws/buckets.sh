#!/usr/bin/env bash
# create bucket when localstack service starts
awslocal s3 mb s3://$S3_BUCKET
