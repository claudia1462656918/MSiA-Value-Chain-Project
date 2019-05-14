from boto3.session import Session
import boto3

ACCESS_KEY = 'AKIAIEYVV2LL6SUCQEHQ'
SECRET_KEY = '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'

bucket_name = 'value-chain-project-data'
data_path = '/Users/xulinxin/Desktop/data.csv'
folder_name = 'value-chain-project'

# download file from s3 data bucket
def get_data(ACCESS_KEY,SECRET_KEY, bucket_name, data_path):
    session = Session(aws_access_key_id=ACCESS_KEY,
              aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')
    your_bucket = s3.Bucket(bucket_name)
    your_bucket.download_file('bank.csv',data_path)

