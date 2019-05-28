import argparse
import boto3
s3 = boto3.client("s3")

def get_data(args):
    s3.download_file(args.bucket_name, args.file_key, args.output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from S3")

    # add argument
    parser.add_argument("--file_key", help="Name of the file in S3 that you want to download")
    parser.add_argument("--bucket_name", help="s3 bucket name")
    parser.add_argument("--output_file_path", help="output path for downloaded file")

    args = parser.parse_args()
    get_data(args)