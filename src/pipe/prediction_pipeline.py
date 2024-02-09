import os
import sys

import torch
import torchaudio

from src.cloud_storage.s3_operations import S3Sync
from src.constants import *
from src.entity.config_entity import PredictionPipelineConfig
from src.exception import CustomException
from src.logger import logging
from src.model.final_model import CNNNetwork
from src.utils import load_object


class SinglePrediction:
    def __init__(self, prediction_pipeline_config: PredictionPipelineConfig,
                s3sync: S3Sync):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.s3_sync = s3sync
        except Exception as e:
            raise CustomException(e, sys)
    
    def _get_model_in_production(self):
        """
        It checks if the model is available in the s3 bucket, if available, it downloads it to the local machine
        and returns the path to the model
        
        Returns:
          The path to the model.
        """
        try:
            s3_model_path = self.prediction_pipeline_config.s3_model_path
            model_download_path = self.prediction_pipeline_config.model_download_path
            os.makedirs(model_download_path, exist_ok=True)
            self.s3_sync.sync_folder_from_s3(folder=model_download_path, aws_bucket_url=s3_model_path)
            for file in os.listdir(model_download_path):
                if file.endswith(".pt"):
                    prediction_model_path = os.path.join(model_download_path, file)
                    logging.info(f"Production model for prediction found in {prediction_model_path}")
                    break
                else:
                    logging.info("Model is not available in Prediction artifacts")
                    prediction_model_path = None
            return prediction_model_path
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def prediction_step(model, class_mapping, input_signal):
        """
        The function takes in a model, an input signal, and a class mapping. It then runs the model on the
        input signal, and returns the language that corresponds to the predicted label.
        
        Args:
          model: the model that you want to use for prediction
          input_signal: the audio signal that we want to classify
          class_mapping: a dictionary of the form {'language': label}
        
        Returns:
          The language that is being predicted.
        """
        try: 
            model.eval()
            with torch.no_grad():
                prediction = model(input_signal)
                prediction_index = prediction[0].argmax(0)
                logging.info("prediction index: {}".format(prediction_index.item()))
                for language, label in class_mapping.items():
                    if label == prediction_index.item():
                        return language
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model(self):
        """
        It loads the model from the path and returns the model object for prediction.
        
        Returns:
          A model object
        """
        try: 
            prediction_model_path = self._get_model_in_production()
            if prediction_model_path is None:
                return None
            else:
                num_classes = NUM_CLASSES
                in_channels = IN_CHANNELS
                prediction_model = CNNNetwork(in_channels=in_channels, num_classes=num_classes)
                model_state_dict = torch.load(prediction_model_path, map_location='cpu')
                prediction_model.load_state_dict(model_state_dict['model_state_dict'])
                prediction_model.eval()
            return prediction_model
        except Exception as e:
            raise CustomException(e, sys)

    def predict_language(self, input_signal):
        """
        It downloads the model and the class mappings from S3, and then uses the model to predict the
        language of the input signal
        
        Args:
          input_signal: The input signal is the text that you want to predict the language for.
        
        Returns:
            output: str
        """
        try: 
            prediction_model = self.get_model()
            os.makedirs(self.prediction_pipeline_config.prediction_artifact_dir, exist_ok=True)
            download_path = self.prediction_pipeline_config.transformation_download_path
            os.makedirs(download_path, exist_ok=True)
            self.s3_sync.sync_folder_from_s3(folder=download_path, aws_bucket_url=S3_ARTIFACTS_URI)
            class_mappings_path = os.path.join(download_path,CLASS_MAPPINGS_OBJECT_NAME)
            class_mapping = load_object(file_path=class_mappings_path)
            if prediction_model is not None:
                output = self.prediction_step(prediction_model, class_mapping, input_signal)
                return output
            else:
                raise CustomException("Model not Found in production", sys)
        except Exception as e:
            raise CustomException(e, sys)
    
# custom dataset for prediction
class LanguageData:
    def __init__(self, transformation, sample_rate, num_samples):
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples
        self.transformation = transformation
    
    def resample_audio(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def mix_down_channels(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim = 0, keepdim=True)
        return signal
    
    def cut_if_needed(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal 
    
    def right_padding(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing = self.num_samples - length_signal
            last_dim_padding = (0, num_missing)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def load_data(self, sample_path):
        signal, sr = torchaudio.load(sample_path)
        signal = self.resample_audio(signal, sr)
        signal = self.mix_down_channels(signal)
        signal = self.cut_if_needed(signal)
        signal = self.right_padding(signal)
        signal = self.transformation(signal)
        return signal
            
                        