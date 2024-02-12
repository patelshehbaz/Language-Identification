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

@dataclass
class DataPreprocessingConfig:
    data_preprocessing_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_PREPROCESSING_ARTIFACTS_DIR)
    metadata_dir_path: str = os.path.join(data_preprocessing_artifacts_dir, METADATA_DIR)
    metadata_path: str = os.path.join(data_preprocessing_artifacts_dir, METADATA_DIR, METADATA_FILE_NAME)
    train_dir_path: str = os.path.join(data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TRAIN_DIR)
    train_file_path: str = os.path.join(data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TRAIN_DIR, TRAIN_FILE_NAME)
    test_dir_path: str = os.path.join(data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TEST_DIR)
    test_file_path: str = os.path.join(data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TEST_DIR, TEST_FILE_NAME)
    transformations_dir: str = os.path.join(data_preprocessing_artifacts_dir, OTHER_ARTIFACTS)
    transformations_object_path = os.path.join(data_preprocessing_artifacts_dir, transformations_dir, TRANSFORMATION_OBJECT_NAME)
    class_mappings_object_path = os.path.join(data_preprocessing_artifacts_dir, transformations_dir, CLASS_MAPPINGS_OBJECT_NAME)
    sample_rate: int = SAMPLE_RATE

@dataclass
class ModelTrainerConfig:
    model_trainer_artifacts_dir :str = os.path.join(from_root(), training_pipeline_config.artifact_dir, MODEL_TRAINING_ARTIFACTS_DIR)
    trained_model_dir: str = os.path.join(model_trainer_artifacts_dir, TRAINED_MODEL_NAME)
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    stepsize: int = STEP_SIZE
    gamma: float = GAMMA

@dataclass
class ModelEvaluationConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    model_evaluation_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR)
    best_model_dir: str = os.path.join(model_evaluation_artifacts_dir, S3_MODEL_DIR_NAME)
    base_accuracy: float = BASE_ACCURACY

@dataclass
class PredictionPipelineConfig:
    s3_model_path = S3_BUCKET_MODEL_URI
    prediction_artifact_dir = os.path.join(from_root(), 'artifact', 'prediction_artifact')
    model_download_path = os.path.join(prediction_artifact_dir, PREDICTION_MODEL_DIR_NAME) 
    transformation_download_path = os.path.join(prediction_artifact_dir, TRANSFORMATION_DOWNLOAD_DIR)
    app_artifacts = os.path.join(prediction_artifact_dir, 'user_input_audio')
    input_sounds_path = os.path.join(app_artifacts, 'inputSound.wav')
