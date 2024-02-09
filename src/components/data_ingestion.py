from src.entity.config_entity import DataIngestionConfig
from src.cloud_storage.s3_operations import S3Sync
from src.exception import CustomException
import sys, os 
from src.logger import logging
from zipfile import ZipFile, Path
from src.constants import S3_BUCKET_URI, UNZIPPED_FOLDERNAME
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig,
                s3_sync: S3Sync):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = s3_sync
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_from_cloud(self) -> None:
        """
        It downloads a zip file from an S3 bucket to a local directory
        
        Returns:
          None
        """
        try:
            logging.info("Initiating data download from s3 bucket...")
            download_dir = self.data_ingestion_config.download_dir
            zip_file_path = self.data_ingestion_config.zip_file_path
            if os.path.isfile(zip_file_path):
                logging.info(
                    f"Data is already present in {download_dir}, So skipping download step.")
                return None
            else:
                self.s3_sync.sync_folder_from_s3(
                    folder=download_dir, aws_bucket_url=S3_BUCKET_URI)
                logging.info(
                    f"Data is downloaded from s3 bucket to Download directory: {download_dir}.")
        except Exception as e:
            raise CustomException(e, sys)
        
    def unzip_data(self) -> Path:
        """
        It unzips the downloaded zip file from the download directory and returns the unzipped folder path
        
        Returns:
          The unzipped data directory path.
        """
        try:
            logging.info(
                "Unzipping the downloaded zip file from download directory...")
            zip_file_path = self.data_ingestion_config.zip_file_path
            unzip_data_path = self.data_ingestion_config.unzip_data_dir_path
            extracted_data_dir = os.path.join(unzip_data_path, UNZIPPED_FOLDERNAME)
            if os.path.isdir(extracted_data_dir):
                logging.info(
                    "Unzipped Folder already exists in unzip directory, so skipping unzip operation.")
            else:
                os.makedirs(unzip_data_path, exist_ok=True)
                with ZipFile(zip_file_path, 'r') as zip_file_ref:
                    zip_file_ref.extractall(unzip_data_path)
            logging.info(
                f"Unzipped file exists in unzip directory: {unzip_data_path}.")
            return extracted_data_dir
        except Exception as e:
            raise CustomException(e, sys)
    
    def rename(self) -> None:
        """
        It renames all the files in the unzipped folder to a single format
        """
        try:
            logging.info(
                "Renaming files in unzip directory to single format...")
            unzip_data_path = self.data_ingestion_config.unzip_data_dir_path
            extract_dir_path = os.path.join(
                unzip_data_path, UNZIPPED_FOLDERNAME)
            for folder in os.listdir(extract_dir_path):
                class_path = extract_dir_path + '/' + str(folder)
                for count, files in enumerate(os.listdir(class_path)):
                    try:
                        dst = f"{folder}-{str(count)}.wav"
                        src = f"{extract_dir_path}/{folder}/{files}"
                        dst = f"{extract_dir_path}/{folder}/{dst}"
                        os.rename(src, dst)
                    except FileExistsError:
                        pass
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self)-> DataIngestionArtifacts:
        """
        It initiates data ingestion component
        
        Returns:
          DataIngestionArtifacts object
        """
        try:
            self.get_data_from_cloud()
            unzip_data_path = self.unzip_data()
            self.rename()
            data_ingestion_artifact = DataIngestionArtifacts(download_dir_path=self.data_ingestion_config.download_dir,
                                                            extract_dir_path=unzip_data_path)
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

