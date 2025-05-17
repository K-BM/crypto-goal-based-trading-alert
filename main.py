from app.data.fetchers import BinanceDataFetcher
from app.data.processors import DataProcessor
from app.predictors.btc_predictor import BTCPredictor
from app.alerts.notifiers import AlertNotifier
from config import BINANCE_CONFIG, THRESHOLD_PCT
from datetime import datetime

def main():
    # Initialize components
    fetcher = BinanceDataFetcher(BINANCE_CONFIG['api_key'], BINANCE_CONFIG['api_secret'])
    processor = DataProcessor()
    predictor = BTCPredictor(THRESHOLD_PCT)
    # notifier = AlertNotifier(EMAIL_RECEIVER)

    
    # Fetch and process data
    print(f"{datetime.now()} - Fetching data...")
    hourly_data = fetcher.fetch_hourly_data("BTCUSDT", "1 Jan 2020")
    hourly_with_indicators = processor.calculate_hourly_indicators(hourly_data)
    
    # Generate predictions
    print(f"{datetime.now()} - Generating predictions...")
    results_df = predictor.evaluate_multiple_thresholds(hourly_with_indicators)

    # Print summary of all thresholds
    print("\n" + "="*120)
    print("SUMMARY OF ALL THRESHOLDS (PRICE INCREASE PREDICTION)".center(120))
    print("="*120)
    print(results_df.round(2).to_string(index=False, justify='center'))
    print("="*120)
    
    # # Send alert
    # print(f"{datetime.now()} - Sending alerts...")
    # html_content = results_df.to_html()
    # notifier.send_email_alert("BTC Prediction Update", html_content)
    
    # print(f"{datetime.now()} - Process completed")

if __name__ == "__main__":
    main()