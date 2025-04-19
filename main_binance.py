import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations (optional)
import logging
from datetime import datetime, timedelta
import pytz
import warnings
from sklearn.model_selection import train_test_split
from crypto_analysis.data_fetching import get_full_1h_data, add_technical_indicators
from crypto_analysis.preprocessing import normalize_data, create_sequences
from crypto_analysis.model import build_and_train_lstm, predict_next_close
from crypto_analysis.utils import plot_loss
import numpy as np

# Suppress warnings (Optional, adjust depending on the verbosity you want)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to set random seed for reproducibility
def set_random_seed(seed):
    np.random.seed(seed)

# Main execution function
def main():
    # Variables
    symbol = "BTCUSDT"
    start_str = "1 Jan 2015"
    riyadh = pytz.timezone("Asia/Riyadh")
    
    try:
        # Fetch data
        logger.info(f"Fetching data for {symbol} starting from {start_str}.")
        df = get_full_1h_data(symbol=symbol, start_str=start_str, end_time=datetime.now(riyadh))
        
        # Add technical indicators
        logger.info("Adding technical indicators to the data.")
        df = add_technical_indicators(df)
        
        # Clean data: remove rows with missing values
        logger.info("Cleaning the data by removing rows with missing values.")
        df.dropna(inplace=True)
        # df =  df.iloc[:-2]  # Remove the last row 

        # Prepare data for normalization
        df_final = df.copy()
        print(f"Last available close price: {df_final.close.tail(1)}")
        
        # Normalize the data
        logger.info("Normalizing the data.")
        df_final, scaler = normalize_data(df_final, df_final.columns)
        
        # Set time steps and forecast horizon
        time_steps = 24  # Using the last 24 hourly data points
        forecast_horizon = 1  # Predict for the next hour
        
        # Create sequences for the forecast horizon
        logger.info(f"Creating sequences with time_steps={time_steps} and forecast_horizon={forecast_horizon}.")
        X, y = create_sequences(df_final, time_steps, forecast_horizon)
        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
        
        # Split the data into training and testing sets (80% train, 20% test)
        logger.info("Splitting the data into train and test sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        
        # Drop the 'close' column from df_final.columns to get the feature columns
        feature_columns = [col for col in df_final.columns if col != 'close']

        # Build and train the LSTM model
        # Set random seed for reproducibility
        set_random_seed(42)
        logger.info("Building and training the LSTM model.")
        model, history = build_and_train_lstm(X_train, y_train, X_test, y_test, time_steps, feature_columns)
        
        # Plot training loss
        logger.info("Plotting the training loss.")
        plot_loss(history)
        
        # Predict the next close price
        logger.info("Predicting the next close price.")
        predicted_next_hour_close_price = predict_next_close(model, X_test, scaler, df_final.columns)
        
        # Log the predicted close price for the next hour
        predicted_time = df.tail(1).index[0] + timedelta(hours=1)
        logger.info(f"Predicted close price for {predicted_time}: {predicted_next_hour_close_price}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

# Run the main function
if __name__ == "__main__":
    main()
