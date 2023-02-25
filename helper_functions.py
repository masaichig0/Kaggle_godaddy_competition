


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (8, 5), "figure.dpi": 120})
import os
from sklearn.preprocessing import MinMaxScaler

# In[6]:


# Creaet the function to label windowed data
def get_labelled_windows(x, horizon=8):
    """
    Creates labels for windowed dataset
    E.g. if horizon=1;
    Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
    """
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=8, horizon=8):
    """
    Turns a 1D array into a 2D array of sequential labelled windows of window_size with horizon 
    size labels. 
    """
    #1. Create a window of specific window_size (add the horizon on the end for labelling later)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    #2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size+horizon-1)), axis=0).T 
    #print(f'Window indexes: \n{window_indexes, window_indexes.shape}')
    #3. Index on the target array (a time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]
    #print(windowed_array)
    #4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels


# In[7]:


# Split train set and test set. 
def make_train_test_split(windows, labels, split_size=1):
    """
      Splits matching pairs of windows and labels into train and test splits.
  
      Returns:
      train_windows, test_windows, train_labels, test_labels
    """
    split_size = len(windows) - split_size
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


# In[8]:
def get_smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_true)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * np.mean(smap)

def get_vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    
    pos_ind = (y_true!=0)|(y_pred!=0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    
    return 100 * smap



# Create the function to take in model predictions and truth values and return evaluation metrics
def evaluate_preds(y_true, y_pred):
    # Make sure float32 datatype for metric calculations
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various evaluation metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)[0]
    #print(f'MAE: {mae}')
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)[0]
    #print(f'MSE: {mse}')
    rmse = tf.sqrt(mse)
    #print(f'RMSE: {rmse}')
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)[0]
    #print(f'MAPE: {mape}')

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy()}


# In[9]:


# Function to predict the model result
def make_preds(model, input_data):
    """
    Uses model to make predictions input_data.
    """
    forecast = model.predict(input_data)
    return tf.squeeze(forecast) # return 1D array of predictions





# Create a function to plot time series data
def plot_time_series(cfips, timesteps, values, format="-", start=0, end=None, label=None):
    """
    Plots timesteps (a series of points in time) against values (a series of values across timesteps).

    Parameters
    -----------
    timesteps : array of timesteps values
    values : array of values across time
    format : style of plot, default "."
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (similar to start but for the end)
    label : label to show on plot about values
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.title(f'Cfips: {cfips}', fontsize=18)
    plt.xlabel("time", fontsize=14)
    plt.ylabel('MicroBusiness Density', fontsize=14)
    if label:
        plt.legend(fontsize=14) # make label bigger
    plt.grid(True)


# In[ ]:


def train_get_result(data, window_size, horizon, epoch, split_size, cfips):
    c = cfips
    df = data.loc[data.cfips == c]
    last_density = df.microbusiness_density.values[-9]
    last_active = df.active.values[-9]
    
    # Create train dataset
    windows, labels = make_windows(df.microbusiness_density.values, window_size=window_size, horizon=horizon)
    
    # Split trin and test set
    train_windows, test_windows, train_labels, test_labels = make_train_test_split(windows, labels, split_size=split_size)
    
    # create model
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
                layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
                layers.LSTM(128, activation="relu", return_sequences=True),
                layers.LSTM(128, activation="relu", return_sequences=True),
                layers.LSTM(128, activation="relu"),
                layers.Dense(horizon)
            ], name=f'lstm_model_{c}')
    
    model.compile(loss='mae', 
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['mae', 'mse'])

    model.fit(x=train_windows, 
             y=train_labels, 
             epochs=epoch,
             batch_size=256, verbose=0)
    
    
    # Predict test data
    preds = make_preds(model, test_windows)
    
    # Evaluate 
    results = evaluate_preds(test_labels, preds)
    mape = results['mape']
    
   
    
    
    return (c, last_density, last_active, mape, preds)


def subplot_by_category(df, i):
    category = df[df['category'] == i]
    plt.rcParams.update({"figure.figsize": (10, 10), "figure.dpi": 120})
    print(f'Category: {i}\nLength of DataFrame: {len(category)}')
    fig, axs = plt.subplots(3)

    # Plot Price histgram
    axs[0].hist(category['MAPE'], bins=50)
    axs[0].set_title(f"MAPE Category {i}", fontsize=16)
    axs[0].set_xlabel("MAPE", fontsize=14)
    axs[0].set_ylabel("Quantity", fontsize=14)

    # Plot Price/SQFT histgram
    axs[1].hist(category['Density'], bins=50)
    axs[1].set_title(f"Density Category {i}", fontsize=16)
    axs[1].set_xlabel("Density", fontsize=14)
    axs[1].set_ylabel("Quantity", fontsize=14)

    # Plot Area to number of ads
    axs[2].hist(category['Active'], bins=50)
    axs[2].set_title(f"Active Category {i}", fontsize=16)
    axs[2].set_xlabel("Active", fontsize=14)
    axs[2].set_ylabel("Quantity", fontsize=14)

   
    fig.tight_layout()
    plt.show()
    
def train_get_results_multi_variables(train_data, window_size, horizon, epoch, split_size, cfips):
    c = cfips
    df = train_data.loc[train_data.cfips == c].reset_index()[['cfips', 'active', 'microbusiness_density', ]]
    last_density = df.microbusiness_density.values[-(horizon+1)]
    last_active = df.active.values[-(horizon+1)]
    
    
    # add window columns
    for i in range(window_size):
        df[f'microbusiness_density+{i+1}'] = df['microbusiness_density'].shift(periods=i+1)

    # create sequenced label
    windows, labels = make_windows(df['microbusiness_density'].values, window_size=(window_size-horizon)+1, horizon=horizon)

    df = df[window_size:]

    df = df.reset_index().drop('index', axis=1)

    # Set microbusiness_density sequence as label, and rest of the data as feature
    X = df.drop(['cfips', 'microbusiness_density'], axis=1)
    Y = labels
    
    
    # Scale the feature
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    train_data, test_data, train_labels, test_labels = make_train_test_split(X, Y, split_size=split_size)

    tf.random.set_seed(42)

    model = tf.keras.Sequential([
                    layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
                    layers.LSTM(128, activation="relu", return_sequences=True),
                    layers.LSTM(128, activation="relu", return_sequences=True),
                    layers.LSTM(128, activation="relu"),
                    layers.Dense(8)
                ], name=f'lstm_model')

    model.compile(loss='mae', 
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae', 'mse'])

    model.fit(x=train_data, 
              y=train_labels, 
              epochs=epoch,
              batch_size=256, verbose=0)

    preds = make_preds(model, test_data)
    results = evaluate_preds(test_labels, preds)
    mape = results['mape']
        

    return (c, last_density, last_active, mape, preds)

def get_score_for_competition(train_data, result_data, window_size, horizon, epoch, split_size, cfips):
    c = cfips
    df = train_data.loc[train_data.cfips == c].reset_index()[['cfips', 'active', 'microbusiness_density', ]]
    last_density = df.microbusiness_density.values[-(horizon+1)]
    last_active = df.active.values[-(horizon+1)]
    
    
    model = result_data.loc[result_data.Country == c]['category'].values[0]
    
    if model == "lstm":
        # add window columns
        for i in range(window_size):
            df[f'microbusiness_density+{i+1}'] = df['microbusiness_density'].shift(periods=i+1)

        # create sequenced label
        windows, labels = make_windows(df['microbusiness_density'].values, window_size=(window_size-horizon)+1, horizon=horizon)

        df = df[window_size:]

        df = df.reset_index().drop('index', axis=1)

        # Set microbusiness_density sequence as label, and rest of the data as feature
        X = df.drop(['cfips', 'microbusiness_density'], axis=1)
        Y = labels
    
    
        # Scale the feature
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
        train_data, test_data, train_labels, test_labels = make_train_test_split(X, Y, split_size=split_size)
        
        test = tf.expand_dims(train_data[-1], axis=0)

        tf.random.set_seed(42)

        model = tf.keras.Sequential([
                        layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
                        layers.LSTM(128, activation="relu", return_sequences=True),
                        layers.LSTM(128, activation="relu", return_sequences=True),
                        layers.LSTM(128, activation="relu"),
                        layers.Dense(8)
                    ], name=f'lstm_model')

        model.compile(loss='mae', 
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['mae', 'mse'])

        model.fit(x=train_data, 
                  y=train_labels, 
                  epochs=epoch,
                  batch_size=256, verbose=0)

        preds = make_preds(model, test)
        forecast = np.array(preds)
        
    else:
        preds = [last_density]*8
        forecast = np.array(preds)

    return (c, forecast)