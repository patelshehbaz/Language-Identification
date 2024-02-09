from datetime import datetime
from from_root import from_root
import os
from src.constants import *
from dataclasses import dataclass


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# The below code is defining the configuration for the training pipeline.
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = 'src'
    artifact_dir: str = os.path.join(from_root(), 'artifact', TIMESTAMP)
    timestamp: datetime = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    download_dir: str = os.path.join(from_root(), DOWNLOAD_DIR)
    zip_file_path: str = os.path.join(download_dir, ZIPFILE_NAME)
    unzip_data_dir_path: str = os.path.join(download_dir, EXTRACT_DIR)

