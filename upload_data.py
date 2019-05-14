from boto3.session import Session
import boto3

# ACCESS_KEY = 'AKIAIEYVV2LL6SUCQEHQ'
# SECRET_KEY = '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'

# bucket_name = 'value-chain-project-data'
# data_path = '/Users/xulinxin/Desktop/data.csv'
# folder_name = 'value-chain-project'

## upload file to s3 project bucket 
def upload_data(ACCESS_KEY,SECRET_KEY, bucket_name, data_path, folder_name):
    session = Session(aws_access_key_id=ACCESS_KEY,
              aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')
    s3.meta.client.upload_file(data_path, bucket_name, '%s/%s' % (folder_name,'bank.csv'))

