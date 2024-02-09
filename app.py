import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from src.entity.config_entity import PredictionPipelineConfig
from src.pipe.prediction_pipeline import LanguageData, SinglePrediction
from src.pipe.training_pipeline import TrainingPipeline
from src.utils import decodesound
from src.cloud_storage.s3_operations import S3Sync
from src.utils import load_object
from src.constants import *

app = Flask(__name__)
CORS(app)

prediction_config = PredictionPipelineConfig()
s3sync = S3Sync()
predictor = SinglePrediction(prediction_config, s3sync)
transformation_dir = prediction_config.transformation_download_path
os.makedirs(transformation_dir, exist_ok=True)
if len(os.listdir(transformation_dir))== 0:
    s3sync.sync_folder_from_s3(folder= transformation_dir, 
                            aws_bucket_url=S3_ARTIFACTS_URI)

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET'])
@cross_origin()
def train():
    train_pipeline = TrainingPipeline()

    train_pipeline.run_pipeline()


@app.route('/predict', methods=['POST'])
@cross_origin()
def predictroute():
    config = PredictionPipelineConfig()
    os.makedirs(config.prediction_artifact_dir, exist_ok=True)
    mel_spectrogram_path = os.path.join(config.transformation_download_path, TRANSFORMATION_OBJECT_NAME)
    mel_spectrogram = load_object(mel_spectrogram_path)
    input_sounds_path = config.input_sounds_path
    app_artifacts = config.app_artifacts
    os.makedirs(app_artifacts, exist_ok=True)
    if request.method == 'POST':
        base_64 = request.json['sound']
        decodesound(base_64, input_sounds_path)
        signal = LanguageData(transformation=mel_spectrogram,
                                sample_rate=SAMPLE_RATE, 
                                num_samples=NUM_SAMPLES).load_data(input_sounds_path)
        signal.unsqueeze_(0)
        result = predictor.predict_language(input_signal=signal)
        return jsonify({"Result" : result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)