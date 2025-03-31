from pymongo import MongoClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Connect to MongoDB
uri = "mongodb+srv://simone:ciaociao12@cluster0.irvdcqp.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client['Stock']
collection = db['sp500']

# Fetch sp500
documents = collection.find()
# Convert documents to a list of dictionaries
data = [doc for doc in documents]
# Create a DataFrame from the data
data = pd.DataFrame(data)
data.drop('_id', axis=1, inplace=True)
data.drop('Adj Close', axis=1, inplace=True)

collections = ['UNRATE', 'INDPRO', 'PPIACO', 'HOUST', 'PI', 'PCE']
for x in collections:
    collection = db[x]
    documents = collection.find()
    # Convert documents to a list of dictionaries
    temp = [doc for doc in documents]
    # Create a DataFrame from the data
    temp = pd.DataFrame(temp)
    temp.drop('_id', axis=1, inplace=True)

    # Merge the DataFrames based on year and month
    data = pd.merge(data, temp, left_on=data['Date'].dt.to_period('M'), right_on=temp['Date'].dt.to_period('M'))
    # Forward-fill the missing values
    data[f'{x} Value'] = data['Value'].ffill()
    data = data.drop('Value', axis=1)
    data = data.drop(['key_0', 'Date_y'], axis=1)
    data = data.rename(columns={'Date_x': 'Date'})

# Close the MongoDB connection
client.close()

# Extract the 'Close' column
close_column = data['Close']
# Convert the 'Date' column to int
data['Date'] = data['Date'].astype(np.int64)
data.drop('Close', axis=1, inplace=True)

# Normalize the input data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
date_scaler = StandardScaler()
date_scaler.fit(data['Date'].values.reshape(-1, 1))
# Normalize the labels
label_scaler = StandardScaler()
scaled_labels = label_scaler.fit_transform(close_column.values.reshape(-1, 1))

# Set the number of days used for prediction
prediction_days = 128
# Initialize empty lists for training data input and output
x_train = []
y_train = []
# Iterate through the scaled data, starting from the prediction_days index
for x in range(prediction_days, len(scaled_data)):
    # Append the previous 'prediction_days' values to x_train
    x_train.append(scaled_data[x - prediction_days:x])
    # Append the current value to y_train
    y_train.append(scaled_labels[x])
# Convert the x_train and y_train lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15, random_state=10)

# Initialize a sequential model
model = Sequential()
model.add(LSTM(units=16, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(units=16, return_sequences=True))
model.add(LSTM(units=16))
model.add(Dense(units=1))

model.summary()
model.compile(optimizer='adam', loss='mse')

# Define the filepath for saving the best model
checkpoint_filepath = 'best_model.h5'
# Create a ModelCheckpoint callback
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=2)
# Create an EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=2)
# Train the LSTM model with the ModelCheckpoint and EarlyStopping callbacks
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.15, callbacks=[checkpoint, early_stopping])

# Loads the weights
model.load_weights(checkpoint_filepath)
# Evaluate the model on the testing set
test_loss = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)

# Make predictions on the testing set
y_pred = model.predict(x_test)

# Denormalize the predicted values and the target values
y_pred_denormalized = label_scaler.inverse_transform(y_pred)
y_test_denormalized = label_scaler.inverse_transform(y_test)

# Extract the last date for each window in x_test
last_dates = x_test[:, -1, 0]
# Denormalize the last_dates using the date_scaler
denormalized_last_dates = date_scaler.inverse_transform(last_dates.reshape(-1, 1))
# Convert the denormalized last_dates to datetime and add one day
last_dates = pd.to_datetime(denormalized_last_dates.flatten()) + pd.DateOffset(days=1)

# Create a new DataFrame for denormalized data
result_df = pd.DataFrame({'Date': last_dates, 'Prediction': y_pred_denormalized.flatten(), 'Target': y_test_denormalized.flatten()})


# Convert the 'Date' column back to datetime64[ns]
result_df['Date'] = pd.to_datetime(result_df['Date'])
# Sort the result_df DataFrame based on the 'Date' column
result_df.sort_values('Date', inplace=True)

# Plot the denormalized data
plt.plot(result_df['Date'], result_df['Prediction'], label='Prediction')
plt.plot(result_df['Date'], result_df['Target'], label='Target')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Prediction vs Target')
plt.legend()
plt.show()


print()


