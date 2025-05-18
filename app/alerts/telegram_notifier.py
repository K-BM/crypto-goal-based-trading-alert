import requests
import pandas as pd
from config import TELEGRAM_CONFIG

class TelegramNotifier:
    def __init__(self):
        self.token = TELEGRAM_CONFIG['bot_token']
        self.chat_id = TELEGRAM_CONFIG['chat_id']  # Using confirmed ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, text):
        """Simplified reliable sender"""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={  # Using json= instead of params=
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': TELEGRAM_CONFIG['parse_mode'],
                    'disable_web_page_preview': True
                },
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Telegram API Error: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
            return False

    def send_enhanced_prediction(self, results_df, title=None):
        results_df['Timestamp'] = pd.to_datetime(results_df['Timestamp'])  # ← Add this line
        sorted_df = results_df.sort_values('Threshold (%)')
        
        # Find first SELL prediction
        first_sell_idx = sorted_df[sorted_df['Latest_Pred'] == 0].index.min()
        
        # If no SELL signals found, show all
        if pd.isna(first_sell_idx):
            filtered_df = sorted_df
        else:
            filtered_df = sorted_df.loc[:first_sell_idx]
        
        header = f"🚀 <b>{title}</b>" if title else "🚀 <b>BTC Prediction Thresholds</b>"
        message = [header, ""]
        
        for _, row in filtered_df.iterrows():
            emoji = "🔴" if row['Latest_Pred'] == 0 else "🟢"
            message.append(
                f"{emoji} Predicted at {row['Prediction_Time_UTC']} | "  # Changed to use Prediction_Time
                f"Open: ${row['Open_Price']:,.2f} → "
                f"Target: ${row['Target_Price']:,.2f} "
                f"(Confidence: {row['Latest_Proba']:.0%}, "
                f"Move: {int(row['Actual_Move'])} | "
                f"Optimal Threshold: {row['Optimal_Threshold']:.2f}%)"
            )

        message.extend([
            "",
            f"📊 <i>Showing {len(filtered_df)}/{len(results_df)} thresholds</i>",
            f"⏱️ Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
        ])
        
        return self.send_message("\n".join(message))

