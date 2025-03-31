import pandas as pd
import yfinance as yf
from kafka import KafkaProducer
from fredapi import Fred

# Initialize Fred API
fred = Fred(api_key='a79fc0c8bf4e63effeb304267ec33dfc')
# Initialize Kafka producer
producer = KafkaProducer(bootstrap_servers='simone-VirtualBox')

start_date = '1982-04-20'
# Symbols for the economic indicators
symbols = ['UNRATE', 'INDPRO', 'PPIACO', 'HOUST', 'PI', 'PCE']
# Create an empty DataFrame to store the data
economic_data = {}
# Retrieve data for each economic indicator and add it to the DataFrame
for symbol in symbols:
    data = fred.get_series(symbol, start_date)
    data = data.reset_index()
    data = data.rename(columns={"index": "Date", 0: "Value"})
    economic_data[symbol] = data

    # Convert the DataFrame rows to JSON and send them to Kafka
    topic = symbol
    count = 0
    for index, row in economic_data[symbol].iterrows():
        message = row.to_json()  # Convert the row to JSON format
        producer.send(topic, value=message.encode('utf-8'))
        count = count + 1
        if count % 1000 == 0:
            producer.flush()
    producer.flush()

producer.close()

