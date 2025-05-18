from app.data.fetchers import BinanceDataFetcher
from app.data.processors import DataProcessor
from app.predictors.btc_predictor import BTCPredictor
from app.alerts.telegram_notifier import TelegramNotifier
from config import TELEGRAM_CONFIG, BINANCE_CONFIG, THRESHOLD_PCT
from datetime import datetime
import pandas as pd

print(f"⏰ Prediction ran at: {datetime.utcnow()} UTC")

def main():
    # Initialize components
    fetcher = BinanceDataFetcher()
    processor = DataProcessor()
    
    # Create predictors for both directions
    predictor_inc = BTCPredictor(THRESHOLD_PCT, direction='increase')
    predictor_dec = BTCPredictor(THRESHOLD_PCT, direction='decrease')

    # Fetch and process data
    print(f"{datetime.now()} - Fetching data...")
    daily_data = fetcher.fetch_daily_data(
        symbol=BINANCE_CONFIG['symbol'],
        start_str=BINANCE_CONFIG['start_date']
    )

    print(f"{datetime.now()} - Calculating technical indicators...")
    daily_with_indicators = processor.calculate_daily_indicators(daily_data)

    print(f"{datetime.now()} - Evaluating price increases...")
    results_inc = predictor_inc.evaluate_multiple_thresholds(daily_with_indicators)

    print(f"{datetime.now()} - Evaluating price decreases...")
    results_dec = predictor_dec.evaluate_multiple_thresholds(daily_with_indicators)

    # Combine results
    combined_results = pd.concat([results_inc, results_dec])

    # Display results
    print("\n" + "="*120)
    print("COMBINED PREDICTION RESULTS".center(120))
    print("="*120)
    print(combined_results.round(2).to_string(index=False, justify='center'))
    print("="*120)

    # Send results to Telegram
    print(f"{datetime.now()} - Sending results to Telegram...")
    telegram = TelegramNotifier()
    
    # Send separate messages for each direction
    telegram.send_enhanced_prediction(
        results_inc.round(2), 
        title="BTC Price Increase Predictions"
    )
    telegram.send_enhanced_prediction(
        results_dec.round(2),
        title="BTC Price Decrease Predictions"
    )
    
    print(f"{datetime.now()} - Results sent to Telegram")

if __name__ == "__main__":
    main()