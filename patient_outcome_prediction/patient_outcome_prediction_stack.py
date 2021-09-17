from aws_cdk import (
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    core
)
from modeling import POPModeleStack

class PatientOutcomePredictionStack(core.Stack):

    def __init__(self, scope: core.Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        render_lambda = _lambda.Function(
            self, 'render_dashboard',
            runtime=_lambda.Runtime.PYTHON_3_7,
            code=_lambda.Code.from_asset('assets/lambda'),
            handler='render_dashboard.handler'
        )

        bucket = s3.Bucket(
            self, "dashboard_bucket", 
            bucket_name=f"pop-modeling-{self.account}",
            removal_policy=core.RemovalPolicy.DESTROY)

        bucket.grant_read(render_lambda)
        s3deploy.BucketDeployment(
            self, 'deploy_dashboard_files',
            sources=[s3deploy.Source.asset('assets/dashboard')],
            destination_bucket=bucket,
            destination_key_prefix='dashboard',
            )

        s3deploy.BucketDeployment(
            self, 'deploy_modeling_files',
            sources=[s3deploy.Source.asset('assets/modeling')],
            destination_bucket=bucket,
            destination_key_prefix='sagemaker',
            )
        
        sm_notebook = POPModeleStack(self, 'ml-modeling-backend', bucket_name=bucket.bucket_name)

        api = apigw.LambdaRestApi(
            self, 'RWE_Dashboard_CDK',
            rest_api_name='rwe-dashboard-demo',
            handler=render_lambda,
            proxy=False
        )
        integration = apigw.LambdaIntegration(
                render_lambda, 
                proxy=False, 
                passthrough_behavior=apigw.PassthroughBehavior.WHEN_NO_TEMPLATES,
                request_templates={"application/json": f"""{{"user": "$input.params('user')","bucket": "{bucket.bucket_name}" }}"""},
                integration_responses=[{
                    "statusCode": "200",
                    "responseTemplates": {
                        "text/html": "$input.path('$')"
                        },
                    'headers': {
                        "Access-Control-Allow-Origin": "*",
                    },
                    "responseParameters": {
                        "method.response.header.Content-Type": "'text/html'"
                        }
                }]
            )

        index_page = api.root
        index_page.add_method("GET", integration, 
        method_responses=[apigw.MethodResponse(status_code="200", response_parameters={"method.response.header.Content-Type": True})])
        dash = index_page.add_resource("dashboard")
        dash.add_method("GET", integration, 
        method_responses=[apigw.MethodResponse(status_code="200", response_parameters={"method.response.header.Content-Type": True})])