from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time
from decouple import config
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bootstrap_servers = config('KAFKA_BOOTSTRAP_SERVERS', default='localhost:9092').split(',')
topic = config('KAFKA_TOPIC', default='hvac-input')
interval = config('STREAM_INTERVAL', cast=float, default=300)

producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    batch_size=16384,
    linger_ms=5,
    compression_type='gzip'
)

def fetch_real_data():
    return {
        'out_temp': 25.0,
        'out_hum': 60.0,
        'num_people': 10,
        'room_area': 100.0,
        'active_units': 2,
        'hour': 14,
        'day': 1
    }

def stream_real_data():
    while True:
        try:
            row = fetch_real_data()
            producer.send(topic, value=row).get(timeout=10)
            logger.info(f"Sent to Kafka: {row}")
            producer.flush()
        except KafkaError as e:
            logger.error(f"Failed to send: {e}")
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
        time.sleep(interval)

if __name__ == '__main__':
    stream_real_data()