import setuptools


with open("README.md") as fp:
    long_description = fp.read()


setuptools.setup(
    name="patient_outcome_prediction",
    version="0.0.1",

    description="A Patient Outcome Prediction CDK Python app",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="author",

    package_dir={"": "patient_outcome_prediction"},
    packages=setuptools.find_packages(where="patient_outcome_prediction"),

    install_requires=[
        "aws-cdk.core==1.78.0",
        "aws-cdk.aws_iam==1.78.0",
        "aws-cdk.aws_sqs==1.78.0",
        "aws-cdk.aws_sns==1.78.0",
        "aws-cdk.aws_sns_subscriptions==1.78.0",
        "aws-cdk.aws_s3==1.78.0",
    ],

    python_requires=">=3.6",

    classifiers=[
        "Development Status :: 4 - Beta",

        "Intended Audience :: Developers",

        "License :: OSI Approved :: Apache Software License",

        "Programming Language :: JavaScript",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",

        "Topic :: Software Development :: Code Generators",
        "Topic :: Utilities",

        "Typing :: Typed",
    ],
)
