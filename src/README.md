## Here are three files that my QA needs to look at:

* src/get_data.py: download data from public S3 bucket
  + file_key: bank.csv
  + bucket_name: value-chain-project-data
  + output_file_path: output path for the downloaded data ( ../data)
  
* src/upload_data.py: upload data to your own S3 bucket
  + input_file_path: local path for uploaded data
  + bucket_name: the S3 bucket name that we want to put the data to
  + output_file_path: output path for uploaded file on S3

* src/sql/models.py: create database
  + RDS True if you want to create database in RDS else None.
** Created database can be checked in 'src/sql/logfile'
