from aws_cdk import (
    aws_sagemaker as sagemaker,
    aws_iam as iam,
    core
)


class POPModeleStack(core.Construct):

    def __init__(self, scope: core.Construct, construct_id: str, bucket_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self._PREFIX = construct_id

        # Create Role for the SageMaker Backend
        self._service_role = iam.Role(
            self, f'{self._PREFIX}-ServiceRole',
            role_name=f'{self._PREFIX}-ServiceRole',
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal('sagemaker.amazonaws.com'),
                iam.ServicePrincipal('lambda.amazonaws.com')
            )
        )
        self.updateServiceRolePermissions()

                # Create SageMaker instance
        self._notebook_lifecycle = sagemaker.CfnNotebookInstanceLifecycleConfig(
            self, f'{self._PREFIX}-LifeCycleConfig',
            notebook_instance_lifecycle_config_name='pop-config',
            on_start=[sagemaker.CfnNotebookInstanceLifecycleConfig.NotebookInstanceLifecycleHookProperty(
                content=core.Fn.base64(f"""
                    #!/bin/bash
                    set -ex

                    BUCKET="{bucket_name}"
                    PREFIX="sagemaker"

                    cd home/ec2-user/SageMaker
                    aws s3 cp s3://$BUCKET/$PREFIX /home/ec2-user/SageMaker --recursive
                    sudo chown "ec2-user":"ec2-user" /home/ec2-user/SageMaker --recursive
                """)
            )]
        )
        self.notebook = sagemaker.CfnNotebookInstance(
            self, f'{self._PREFIX}-notebook',
            role_arn=self._service_role.role_arn,
            instance_type='ml.t3.medium',
            notebook_instance_name='POP-PatientOutcomePrediction',
            lifecycle_config_name ='pop-config'
        )

    def updateServiceRolePermissions(self):
        # Service Permissions 
        self._service_role.add_to_policy(iam.PolicyStatement(
            effect = iam.Effect.ALLOW,
            resources = ['*'],
            actions = ['*'],
            conditions = [],
        ))

