import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import backend as K

# Set random seeds for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)  # For numpy
    random.seed(seed)  # For python's random module
    tf.random.set_seed(seed)  # For TensorFlow

    # Set deterministic behavior (may slow down training)
    # Disable GPU specific non-deterministic operations
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )

# Disable Dropout during prediction
def disable_dropout(model):
    # Set the dropout layers to zero during inference
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = 0.0
    return model

# Build and train LSTM model
def build_and_train_lstm(X_train, y_train, X_test, y_test, time_steps, features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(time_steps, len(features))))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    model = disable_dropout(model)  # Disable dropout during prediction
    return model, history

# Function to make predictions
def predict_next_close(model, X_test, scaler, features):
    last_time_steps = X_test[-1:].reshape(1, X_test.shape[1], X_test.shape[2])
    next_prediction = model.predict(last_time_steps)
    predicted_scaled_close = next_prediction[0][0]

    # Inverse transform to get original scale
    predicted_close_original = scaler.inverse_transform(
        np.array([[predicted_scaled_close] + [0] * (len(features) - 1)])
    )
    return predicted_close_original[0][0]