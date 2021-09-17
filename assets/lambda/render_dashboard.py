import json
import boto3
def handler(event, context):
    bucket = event["bucket"]
    file_name =  event.get("user", "") 
    if file_name=="":
        key = "index.html"
    else:
        key = file_name+".html"
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key='dashboard/'+key)
    content = response['Body'].read().decode('utf-8')

    return content