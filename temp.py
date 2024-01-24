
def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

datetime_object = str_to_datetime('1986-03-19')
datetime_object

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = pd.to_datetime(first_date_str)
    last_date = pd.to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    while target_date <= last_date:
        if target_date in dataframe.index:
            df_subset = dataframe.loc[:target_date].tail(n+1)

            if len(df_subset) == n+1:
                values = df_subset['Close'].to_numpy()
                x, y = values[:-1], values[-1]

                dates.append(target_date)
                X.append(x)
                Y.append(y)

        # Move to the next date
        target_date += datetime.timedelta(days=1)

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

# Loading the data


# Convert 'Date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
data = df.set_index('Date')

# Use the df_to_windowed_df function after preparing the data
windowed_df = df_to_windowed_df(data, '2001-02-21',
                                '2012-10-23',  n=3)
print(windowed_df)

columns_to_scale = ['Target-3', 'Target-2', 'Target-1', 'Target']  # Replace with your column names

# Create a MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale only the numerical columns
numerical_data = windowed_df[columns_to_scale]
scaled_data = scaler.fit_transform(numerical_data)

# Replace the original numerical columns with the scaled values
windowed_df[columns_to_scale] = scaled_data

# Now, windowed_df contains the scaled values for the specified columns
print(windowed_df)

data = data.dropna()

# Displaying the first few rows of the cleaned DataFrame

data.head()

plt.plot(windowed_df['Target Date'],windowed_df['Target'])



# Convert the DataFrame to a NumPy array
df_as_np = windowed_df.to_numpy()



# Extract dates from the first column
dates = df_as_np[:,0]



# Extract the feature matrix from columns 1 to second-to-last column
middle_matrix = df_as_np[:, 1:-1]

# Reshape the feature matrix
X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

# Extract the target values from the last column
Y = df_as_np[:, -1]

# Convert arrays to np.float32 data type

X = X.astype(np.float32)
y = Y.astype(np.float32)




# Print the shapes of the arrays
print("Dates shape:", dates.shape)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

q_80 = int(len(dates)* .8)
q_90 = int(len(dates)* .9)

dates_train, X_train, y_train = dates[:q_80],X[:q_80],y[:q_80]


dates_val, X_val, y_val = dates[q_80:q_90],X[q_80:q_90],y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:],X[q_90:],y[q_90:]


plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])

with st.spinner("Loading..."):
    t.sleep(10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),

                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

train_predictions = model.predict(X_train).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations'])


val_predictions = model.predict(X_val).flatten()

plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.legend(['Validation Predictions', 'Validation Observations'])

test_predictions = model.predict(X_test).flatten()

plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Testing Predictions', 'Testing Observations'])

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predictions',
            'Training Observations',
            'Validation Predictions',
            'Validation Observations',
            'Testing Predictions',
            'Testing Observations'])

from copy import deepcopy
#test_predictions = model.predict(X_test).flatten()
recursive_predictions = []
recursive_dates = np.concatenate([dates_val, dates_test])
#recursive_dates = np.concatenate([dates_val])
last_window = deepcopy(X_train[-1])  # Start with the last window from the training set
#test_predictions = model.predict(X_test).flatten()
print(last_window)
for target_date in recursive_dates:
#    # Predict the next value based on the current window
    next_prediction = model.predict(np.array([last_window])).flatten()
    recursive_predictions.append(next_prediction)
    print(next_prediction)
    # Shift the window: drop the first element and append the new prediction
    last_window = np.roll(last_window, -1, axis=0)
    last_window[-1] = next_prediction

 #'recursive_predictions' contains the recursively predicted values

import matplotlib.pyplot as plt

# Define colors for each line
colors = {
    'Training Predictions': 'blue',
    'Training Observations': 'green',
    'Validation Predictions': 'orange',
    'Validation Observations': 'red',
    'Testing Predictions': 'purple',
    'Testing Observations': 'brown',
    'Recursive Predictions': 'magenta'
}

# Plot the data with labels and custom styles
plt.figure(figsize=(12, 6))  # Adjust the figure size

for label in colors.keys():
    plt.plot([], [], label=label, color=colors[label])  # Create empty lines for legend
n=2000  #window size for prediction
# Plot the actual data

#plt.plot(dates_train, train_predictions, color=colors['Training Predictions'])
#plt.plot(dates_train, y_train, linestyle='--', color=colors['Training Observations'])
plt.plot(dates_val[:n], val_predictions[:n], color=colors['Validation Predictions'])
plt.plot(dates_val[:n], y_val[:n], linestyle='--', color=colors['Validation Observations'])
#plt.plot(dates_test, test_predictions, color=colors['Testing Predictions'])
#plt.plot(dates_test, y_test, linestyle='--', color=colors['Testing Observations'])
plt.plot(recursive_dates[:n], recursive_predictions[:n], color=colors['Recursive Predictions'])

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Predictions vs. Observations For 1st Model')

# Add a legend with a better layout
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Display grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Add data points to the legend for better identification
handles, labels = plt.gca().get_legend_handles_labels()
labels_and_handles = zip(labels, handles)
handles = [h[1] for h in sorted(labels_and_handles, key=lambda x: x[0])]
labels = [h[0] for h in sorted(labels_and_handles, key=lambda x: x[0])]
plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()  # Adjust spacing
plt.show()


########################################################################################################
#Plotting on site

st.write("First few rows of the cleaned DataFrame:")
st.write(data.head())

st.write("First few rows of the cleaned DataFrame:")
st.write(windowed_df)

st.line_chart(windowed_df.set_index("Target Date"))


# Create a DataFrame for Streamlit line_chart
data = {'dates': dates_train, 'train_predictions': train_predictions, 'y_train': y_train}
df = pd.DataFrame(data).set_index('dates')

# Plotting using Streamlit line_chart
st.line_chart(df)


# Assuming you have dates_train, train_predictions, y_train, dates_val, val_predictions, y_val, dates_test, test_predictions, y_test as your data
# Generate example data for demonstration
dates_train = pd.date_range('2023-01-01', '2023-01-10')
train_predictions = np.random.rand(len(dates_train))
y_train = np.random.rand(len(dates_train))

dates_val = pd.date_range('2023-01-11', '2023-01-15')
val_predictions = np.random.rand(len(dates_val))
y_val = np.random.rand(len(dates_val))

dates_test = pd.date_range('2023-01-16', '2023-01-20')
test_predictions = np.random.rand(len(dates_test))
y_test = np.random.rand(len(dates_test))

# Create a DataFrame for Streamlit line_chart
data = {
    'Date': np.concatenate([dates_train, dates_val, dates_test]),
    'Training Predictions': np.concatenate([train_predictions, [np.nan] * (len(dates_val) + len(dates_test))]),
    'Training Observations': np.concatenate([y_train, [np.nan] * (len(dates_val) + len(dates_test))]),
    'Validation Predictions': np.concatenate([[np.nan] * len(dates_train), val_predictions, [np.nan] * len(dates_test)]),
    'Validation Observations': np.concatenate([[np.nan] * len(dates_train), y_val, [np.nan] * len(dates_test)]),
    'Testing Predictions': np.concatenate([[np.nan] * (len(dates_train) + len(dates_val)), test_predictions]),
    'Testing Observations': np.concatenate([[np.nan] * (len(dates_train) + len(dates_val)), y_test]),
}

df = pd.DataFrame(data).set_index('Date')

# Plotting using Streamlit line_chart
st.line_chart(df)

# Add legend to the plot
st.pyplot(plt)




st.area_chart(df)

st.bar_chart(df)

import plotly.express as px
fig = px.line(df, x=df.index, y=df.columns, title='Stock Price Predictions')
st.plotly_chart(fig)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
df.plot(y=['Training Predictions', 'Training Observations'], ax=axes[0])
df.plot(y=['Validation Predictions', 'Validation Observations'], ax=axes[1])
df.plot(y=['Testing Predictions', 'Testing Observations'], ax=axes[2])
st.pyplot(fig)






st.header("Machine Learning")

#sub Header 
st.subheader("Linear Regression")

#
st.info("Information details of a user")

#Warning
st.warning("ERORR:404")

#
st.write("Employee Name")
st.write("Employee Name")

#
st.markdown("# Hello")
st.markdown("## Hello")

#
st.text("Hello im here")

#
st.caption("Caption is here")

#
st.latex(r''' a+b x^2+c''')


st.text("Copyright - All Rights Reserved")

