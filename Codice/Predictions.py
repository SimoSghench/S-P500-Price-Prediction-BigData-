from keras.models import load_model
from pymongo import MongoClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Connect to MongoDB
uri = "mongodb+srv://simone:ciaociao12@cluster0.irvdcqp.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client['Stock']
collection = db['sp500']

# Fetch sp500
documents = collection.find()
# Convert documents to a list of dictionaries
initial_data = [doc for doc in documents]
# Create a DataFrame from the data
initial_data = pd.DataFrame(initial_data)
initial_data.drop('_id', axis=1, inplace=True)
initial_data.drop('Adj Close', axis=1, inplace=True)

data = initial_data
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

# UNRATE, INDPRO,...,PCE values stops in MAY so I replicate them.
# Filter the relevant data from initial_data
filtered_initial_data = initial_data[initial_data['Date'] > data["Date"].iloc[-1]]
# Get the last values of the columns to replicate in data
last_values = data.iloc[-1][['UNRATE Value', 'INDPRO Value', 'PPIACO Value', 'HOUST Value', 'PI Value', 'PCE Value']]
# Append the filtered initial_data to data
data = pd.concat([data, filtered_initial_data[['Date', 'High', 'Volume', 'Low', 'Close', 'Open']]], ignore_index=True)
# Fill the NaN values in the additional columns with the last values
data[['UNRATE Value', 'INDPRO Value', 'PPIACO Value', 'HOUST Value', 'PI Value', 'PCE Value']] = \
    data[['UNRATE Value', 'INDPRO Value', 'PPIACO Value', 'HOUST Value', 'PI Value', 'PCE Value']].fillna(last_values)

# Dataframe containing all the prediction and the actual value
df_pred = pd.DataFrame()
df_pred["Date"] = data["Date"]
df_pred["Value"] = data['Close']
# Preparing the dataframe to receive the predictions
# Get the last date from the existing data
last_date = df_pred["Date"].iloc[-1]
# Add three new rows to df_pred
df_pred.loc[len(df_pred)] = [last_date + pd.DateOffset(days=1), np.nan]
df_pred.loc[len(df_pred)] = [last_date + pd.DateOffset(days=2), np.nan]
df_pred.loc[len(df_pred)] = [last_date + pd.DateOffset(days=3), np.nan]
df_pred["Value"].iloc[-3] = np.nan

# Extract the 'Close' column
close_column = data['Close']
# Convert the 'Date' column to int
data['Date'] = data['Date'].astype(np.int64)
data.drop('Close', axis=1, inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
date_scaler = StandardScaler()
date_scaler.fit(data['Date'].values.reshape(-1, 1))
# Normalize the labels
label_scaler = StandardScaler()
scaled_labels = label_scaler.fit_transform(close_column.values.reshape(-1, 1))
# Set the number of days used for prediction
prediction_days = 128

for i in range(1, 4):
    x_test = []
    # Iterate through the scaled data, starting from the prediction_days index
    for x in range(prediction_days, len(scaled_data)+1):
        # Append the previous 'prediction_days' values to x_train
        x_test.append(scaled_data[x - prediction_days:x])

    # Convert the x_test and y_test lists to numpy arrays
    x_test = np.array(x_test)

    # Load the model from the saved weights
    loaded_model = load_model(f'best_model{i}.h5')
    # Perform new predictions on new data
    y_pred = loaded_model.predict(x_test)

    # Denormalize the predicted values
    y_pred_denormalized = label_scaler.inverse_transform(y_pred)
    # Extract the last date for each window in x_test
    last_dates = x_test[:, -1, 0]
    # Denormalize the last_dates using the date_scaler
    denormalized_last_dates = date_scaler.inverse_transform(last_dates.reshape(-1, 1))
    # Convert the denormalized last_dates to datetime and add i day
    last_dates = pd.to_datetime(denormalized_last_dates.flatten()) + pd.DateOffset(days=i)

    # Create the 'temp' DataFrame
    temp = pd.DataFrame({f'Pred{i}': y_pred_denormalized.flatten()})
    # Create a new DataFrame with NaN values
    nan_values = pd.DataFrame({f'Pred{i}': [np.nan] * (127 + i)})
    # Concatenate the new DataFrame with the temp DataFrame
    temp = pd.concat([nan_values, temp], ignore_index=True)
    # Merge 'temp' with 'df_pred' based on their index
    df_pred[f'Pred{i}'] = temp[f'Pred{i}']

# Save predictions in my database
collection = db['sp500_pred']
# Collect documents as a list of dictionaries
documents = df_pred.to_dict(orient='records')
# Insert the documents into the collection
collection.insert_many(documents)

# Close the MongoDB connection
client.close()

