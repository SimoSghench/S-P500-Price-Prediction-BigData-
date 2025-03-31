from kafka import KafkaConsumer
import json
from datetime import datetime
from pymongo import MongoClient

uri = "mongodb+srv://simone:ciaociao12@cluster0.irvdcqp.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri)
db = client['Stock']
collection = db['sp500']

consumer = KafkaConsumer(
    'sp500',
    bootstrap_servers=['simone-VirtualBox']
)

# Consume messages
for message in consumer:
    message_value = message.value.decode('utf-8')
    data = json.loads(message_value)

    # Convert the date field back to a datetime object in UTC
    timestamp = int(data['Date']) / 1000
    date = datetime.utcfromtimestamp(timestamp).replace(tzinfo=None)
    data['Date'] = date
    # Insert the document into MongoDB
    collection.insert_one(data)

# Close the Kafka consumer and MongoDB client
consumer.close()
client.close()

