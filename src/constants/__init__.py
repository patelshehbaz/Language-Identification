# common constants
DOWNLOAD_DIR = 'downloads'
EXTRACT_DIR = 'downloaded_language_data'
ZIPFILE_NAME = 'language-audio-data.zip'
S3_BUCKET_URI = 's3://languagee-audio-data/dataset/'
UNZIPPED_FOLDERNAME = 'language-audio-data'

DATA_PREPROCESSING_ARTIFACTS_DIR = 'data_preprocessing_artifacts'
METADATA_DIR = 'metadata'
METADATA_FILE_NAME: str = "metadata.csv"
TRAIN_FILE_NAME: str = "metadata_train.csv"
TEST_FILE_NAME: str = "metadata_test.csv"

# constants related to data preprocessing
DATA_PREPROCESSING_TRAIN_DIR: str = "train"
DATA_PREPROCESSING_TEST_DIR: str = "test"
DATA_PREPROCESSING_TRAIN_TEST_SPLIT_RATION: float = 0.3
OTHER_ARTIFACTS = 'transformation'
TRANSFORMATION_OBJECT_NAME = 'mel_spectrogram.pkl'
CLASS_MAPPINGS_OBJECT_NAME = 'class_mappings.pkl'
S3_ARTIFACTS_URI: str = "s3://languagee-audio-data/transformation-artifacts/"

# constants related to data transformations
SAMPLE_RATE: int = 4000
NUM_SAMPLES: int = 20000
FFT_SIZE: int = 1024
HOP_LENGTH: int = 512
N_MELS: int = 64

# constants related to model training
MODEL_TRAINING_ARTIFACTS_DIR: str = "model_training_artifacts"
TRAINED_MODEL_NAME = 'model.pt'
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 128
NUM_WORKERS = 0
STEP_SIZE = 6
GAMMA = 0.5

# constants related to model evaluation
S3_BUCKET_MODEL_URI: str = "s3://languagee-audio-data/model/"
MODEL_EVALUATION_DIR: str = "model_evaluation"
S3_MODEL_DIR_NAME: str = "s3_model"
IN_CHANNELS: int = 1
BASE_ACCURACY: float = 0.6

#constants related to predction pipeline
PREDICTION_MODEL_DIR_NAME = "prediction_model"
TRANSFORMATION_DOWNLOAD_DIR = "transformations"
NUM_CLASSES = 4