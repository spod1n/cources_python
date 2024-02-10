import boto3

s3 = boto3.client('s3')

bucket_name = 'my-bucket'
s3.create_bucket(Bucket=bucket_name)

file_path = 'text.txt'
object_key = f'f_txt/{file_path}'

with open(file_path, 'rb') as file:
    s3.upload_fileobj(file, bucket_name, object_key)

download_path = 'download_text.txt'

with open(download_path, 'wd') as file:
    s3.download_fileobj(bucket_name, object_key, file)

response = s3.list_objects_v2(Bucket=bucket_name)
for obj in response['Contents']:
    print('Object Key: ', obj['Key'])
