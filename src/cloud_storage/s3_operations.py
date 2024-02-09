import os

# It's a class that uses the aws s3 sync command to sync a folder to and from an s3 bucket
class S3Sync:
    def sync_folder_to_s3(self,folder,aws_bucket_url):
        """
        It takes a local folder and an AWS bucket URL as input, and then it uploads the folder to the s3 bucket
        
        Args:
          folder: The folder you want to sync to S3
          aws_bucket_url: s3://bucket-name/folder-name
        """
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        """
        It takes a local folder and an s3 bucket url as input and downloads the files from the s3 bucket
        
        Args:
          folder: The folder you want to sync to S3
          aws_bucket_url: s3://bucket-name/folder-name
        """
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        os.system(command)