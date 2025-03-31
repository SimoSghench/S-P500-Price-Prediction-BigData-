import datetime
import streamlit as st
import pymongo
import pandas as pd
import matplotlib.pyplot as plt

# Connect to MongoDB
uri = "mongodb+srv://simone:ciaociao12@cluster0.irvdcqp.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(uri)
db = client['Stock']
collection = db['sp500_pred']


@st.cache_data
def load_data():
    # Retrieve all documents from the MongoDB collection
    data = list(collection.find())

    # Convert the MongoDB documents to a pandas DataFrame
    df = pd.DataFrame(data)
    df.drop("_id", axis=1, inplace=True)
    # Set the "Date" field as the DataFrame index
    df.set_index("Date", inplace=True)
    return df


def filter_data(df, start_date, end_date):
    # Filter the DataFrame based on the selected date range
    filtered_df = df.loc[start_date:end_date]
    return filtered_df


# Set the title of the Streamlit app
st.title("S&P 500 Price Dashboard")

# Load all data from MongoDB collection
df = load_data()

# Add date range selection sidebar
st.sidebar.title("Date Range Selection")
default_start_date = datetime.datetime(2023, 5, 1)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date")

# Add checkbox selection for different values
show_value = st.sidebar.checkbox("Price", value=True)
show_pred1 = st.sidebar.checkbox("Prediction 1")
show_pred2 = st.sidebar.checkbox("Prediction 2")
show_pred3 = st.sidebar.checkbox("Prediction 3")

# Filter data based on the selected date range
filtered_df = filter_data(df, start_date, end_date)
# Rearrange column order
filtered_df = filtered_df[["Value", "Pred1", "Pred2", "Pred3"]]

selected_rows = df.iloc[[-4, -3, -2, -1]]
# Write the selected rows
date_format = "%Y-%m-%d"
st.header("Price and predictions relative to the last known day")
st.header(f"Date: {selected_rows.index[-4].strftime(date_format)}")
st.header(f"Value: {round(selected_rows['Value'].iloc[0], 2)}")
st.header(f"Next day prediction: {round(selected_rows['Pred1'].iloc[1], 2)}")
st.header(f"Two days prediction: {round(selected_rows['Pred2'].iloc[2], 2)}")
st.header(f"Three days prediction: {round(selected_rows['Pred3'].iloc[3], 2)}")

# Plot the selected values over time
fig, ax = plt.subplots()
if show_value:
    ax.plot(filtered_df["Value"], label="Value")
if show_pred1:
    ax.plot(filtered_df["Pred1"], label="Prediction 1")
if show_pred2:
    ax.plot(filtered_df["Pred2"], label="Prediction 2")
if show_pred3:
    ax.plot(filtered_df["Pred3"], label="Prediction 3")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title("Stock Price Comparison")
plt.xticks(rotation=45)
ax.legend()
st.pyplot(fig)

# Display the filtered data in the Streamlit app
st.write(filtered_df)


