import os
import pandas as pd
import joblib
from aiokafka import AIOKafkaConsumer
import asyncio
import logging
from prometheus_client import Counter, start_http_server
from decouple import config
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

predictions_total = Counter('hvac_predictions_total', 'Total HVAC predictions made')

# Dynamic model path
IN_DOCKER = os.getenv('IN_DOCKER', 'false').lower() == 'true'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = '/app/models' if IN_DOCKER else os.path.join(BASE_DIR, '..', 'models')

# Load models
try:
    logger.info(f"Loading models from {MODELS_DIR}")
    clf_model = joblib.load(os.path.join(MODELS_DIR, 'best_rf_classifier.pkl'))
    reg_model = joblib.load(os.path.join(MODELS_DIR, 'best_rf_regressor.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise

class HVACModel(BaseEstimator):
    def __init__(self, reg_model, clf_model):
        self.reg_model = reg_model
        self.clf_model = clf_model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        clf_pred = self.clf_model.predict(X)
        reg_pred = self.reg_model.predict(X)
        return clf_pred, reg_pred

    def __sklearn_is_fitted__(self):
        return True

pipeline = Pipeline([('scaler', scaler), ('hvac_model', HVACModel(reg_model, clf_model))])

async def kafka_consumer():
    consumer = AIOKafkaConsumer(
        config('KAFKA_TOPIC', default='hvac-input'),
        bootstrap_servers=config('KAFKA_BOOTSTRAP_SERVERS', default='localhost:9092').split(','),
        group_id='hvac-group',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    await consumer.start()
    try:
        while True:
            batch = await consumer.getmany(timeout_ms=1000, max_records=10)
            for topic_partition, messages in batch.items():
                rows = [msg.value for msg in messages]
                if rows:
                    df = pd.DataFrame(rows)
                    required_columns = ['out_temp', 'out_hum', 'num_people', 'room_area', 'active_units', 'hour', 'day', 'prev_setpoint', 'prev_fan_speed', 'prev_out_temp']
                    if not all(col in df.columns for col in required_columns):
                        logger.error("Missing features in Kafka message")
                        continue
                    X = df[required_columns]
                    fan_speed_pred, setpoint_pred = pipeline.predict(X)
                    for i, (sp, fs) in enumerate(zip(setpoint_pred, fan_speed_pred)):
                        predictions_total.inc()
                        logger.info(f"Prediction - Setpoint: {sp}, Fan Speed: {fs}")
    finally:
        await consumer.stop()

if __name__ == '__main__':
    start_http_server(8001)
    asyncio.run(kafka_consumer())