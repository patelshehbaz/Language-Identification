import torch
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.custom_dataset import IndianLanguageDataset
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import *
from src.entity.config_entity import *
from src.exception import CustomException
from src.logger import logging
from src.model.final_model import CNNNetwork
from src.cloud_storage.s3_operations import S3Sync


class TrainingPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config, s3_sync=S3Sync())
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Data ingestion step completed successfully in train pipeline")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_preprocessing(self, data_ingestion_artifacts) -> DataPreprocessingArtifacts:
        logging.info("Starting data preprocessing in training pipeline")
        try:
            data_preprocessing = DataPreprocessing(data_preprocessing_config=self.data_preprocessing_config,
                                                   data_ingestion_artifacts=data_ingestion_artifacts)
            data_preprocessing_artifacts = data_preprocessing.initiate_data_preprocessing()
            logging.info(
                "Data preprocessing step completed successfully in train pipeline")
            return data_preprocessing_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_training(self, data_preprocessing_artifacts: DataPreprocessingArtifacts) -> ModelTrainerArtifacts:
        logging.info("Starting model training in training pipeline")
        try:
            logging.info(
                "Instantiating train and validation dataset from custom dataset class...")
            train_metadata = data_preprocessing_artifacts.train_metadata_path
            test_metadata = data_preprocessing_artifacts.test_metadata_path
            audio_dir = os.path.join(self.data_ingestion_config.unzip_data_dir_path, UNZIPPED_FOLDERNAME)
            target_sample_rate = self.data_preprocessing_config.sample_rate
            transformation_obj = data_preprocessing_artifacts.transformation_object
            train_data = IndianLanguageDataset(metadata=train_metadata, 
                                            audio_dir=audio_dir, 
                                            target_sample_rate=target_sample_rate, 
                                            num_samples=NUM_SAMPLES, 
                                            transformation=transformation_obj)

            test_data = IndianLanguageDataset(metadata=test_metadata, 
                                            audio_dir=audio_dir, 
                                            target_sample_rate=target_sample_rate, 
                                            num_samples=NUM_SAMPLES, 
                                            transformation=transformation_obj)    

            logging.info("Instantiating CNNNetwork model...")
            model = CNNNetwork(
                in_channels=1, num_classes=data_preprocessing_artifacts.num_classes)

            logging.info("Instantiating model trainer class...")
            model_trainer = ModelTrainer(modeltrainer_config=self.model_trainer_config,
                                         model=model,
                                         train_data=train_data,
                                         test_data=test_data,
                                         optimizer_func=torch.optim.Adam)

            logging.info(
                f"The training pipeline is current running in device: {model_trainer.device}.")
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info(
                "Model trainer step completed successfully in train pipeline")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_evaluation(self, model_trainer_artifacts):
        logging.info("Starting model evaluation in training pipeline")
        try:
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                               model_trainer_artifacts=model_trainer_artifacts)
            logging.info("Evaluating current trained model")
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info(
                "Model evaluation step completed successfully in train pipeline")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_pusher(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        logging.info("Starting model pusher in training pipeline")
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifacts=model_evaluation_artifacts)
            logging.info(
                "If model is accepted in model evaluation. Pushing the model into production storage")
            model_pusher_artifacts = model_pusher.initiate_model_pusher()
            logging.info(
                "Model pusher step completed successfully in train pipeline")
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self) -> None:
        """
        The function runs the data ingestion, data preprocessing, model training, model evaluation, and
        model pusher steps in the pipeline and completes the training pipeline
        """
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_preprocessing_artifacts = self.start_data_preprocessing(
                data_ingestion_artifacts=data_ingestion_artifacts)

            model_trainer_artifacts = self.start_model_training(
                data_preprocessing_artifacts=data_preprocessing_artifacts)

            model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts)

            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifacts=model_evaluation_artifacts)

            logging.info(f" The final model pusher artifacts {model_pusher_artifact}")

            logging.info("<<<< Training pipeline completed >>>>")
        except Exception as e:
            raise CustomException(e, sys)

    

