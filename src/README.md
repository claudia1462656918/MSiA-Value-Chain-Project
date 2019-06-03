## Here are three files that my QA（Ziying Wang) needs to look at:

* src/get_data.py: download data from public S3 bucket
  + file_name: bank.csv
  + sourceurl: https://value-chain-project-data.s3.us-east-2.amazonaws.com/bank.csv
  + save_path: output path for the downloaded data ( data/sample/bank.csv)
  
* src/upload_data.py: upload data to your own S3 bucket
  + input_file_path: local path for uploaded data
  + bucket_name: the S3 bucket name that we want to put the data to
  + output_file_path: output path for uploaded file on S3
  

* src/model.py: create database
  + RDS True if you want to create database in RDS else None.
** Created database can be checked in 'src/sql/logfile'
