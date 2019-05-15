

import argparse
import boto3
s3 = boto3.client("s3")

def upload_data(args):
	session = Session(aws_access_key_id=args.aws_access_key_id,
              aws_secret_access_key=args.aws_secret_access_key)
    s3 = session.resource('s3')
    s3.meta.client.upload_file(args.input_file_path, args.bucket_name, args.output_file_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload data to S3")

    # add argument
    parser.add_argument("--input_file_path", help="local path for uploaded file")
    parser.add_argument("--bucket_name", help="s3 bucket name")
    parser.add_argument("--output_file_path", help="output path for uploaded file")

    args = parser.parse_args()
    upload_data(args)


