import yfinance as yf
from kafka import KafkaProducer

# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='simone-VirtualBox')

# Retrieve S&P 500 data
data = yf.download("^GSPC", start="1982-04-20")  # before this date the opening price is 0

# Reset index to numeric index
data.reset_index(inplace=True)

# Convert the DataFrame rows to JSON and send them to Kafka
topic = 'sp500'
count = 0
for index, row in data.iterrows():
    message = row.to_json()  # Convert the row to JSON format
    producer.send(topic, value=message.encode('utf-8'))
    count = count + 1
    if count % 1000 == 0:
        producer.flush()
producer.flush()

producer.close()



