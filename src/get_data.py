import requests
import argparse
import logging

logger = logging.getLogger(__name__)

def download_data(args):
    """
    Downloads data from given public bucket url to the local folder given the path
    
    :param sourceurl (str): url of the stored data in s3 bucket
    :param filename (str): name of the target file
    :param savename (str): save path of the file

    Return: None
    """
    try:
        r = requests.get(args.sourceurl)
        logger.info("Download %s from bucket %s", args.filename, args.sourceurl)
        open(args.savename, 'wb').write(r.content)
    except requests.exceptions.RequestException:
        logger.error("Error: Unable to download file %s", args.filename)

if __name__ == "__main__":
    #download data from the source
    parser = argparse.ArgumentParser(description="Download data from S3")

    parser.add_argument("--sourceurl", help="Target S3 bucket name url")
    parser.add_argument("--filename", help="Target file want to dowlaod")
    parser.add_argument("--savename", help="Filename to be save")

    args = parser.parse_args()
    download_data(args)

