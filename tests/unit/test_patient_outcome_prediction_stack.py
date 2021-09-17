import json
import pytest

from aws_cdk import core
from patient_outcome_prediction.patient_outcome_prediction_stack import PatientOutcomePredictionStack


def get_template():
    app = core.App()
    PatientOutcomePredictionStack(app, "patient-outcome-prediction")
    return json.dumps(app.synth().get_stack("patient-outcome-prediction").template)


def test_role_created():
    assert("AWS::IAM::Role" in get_template())

def test_sagemaker_created():
    assert("AWS::S3::Bucket" in get_template())

def test_s3_created():
    assert("AWS::SageMaker::NotebookInstance" in get_template())

def test_lambda_created():
    assert("AWS::Lambda::Function" in get_template())

def test_apigw_created():
    assert("AWS::ApiGateway::RestApi" in get_template())

def test_aptgw_deploy():
    assert("AWS::ApiGateway::Deployment" in get_template())

def test_apigw_stage():
    assert("AWS::ApiGateway::Stage" in get_template())