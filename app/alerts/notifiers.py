import smtplib
from email.mime.text import MIMEText
from config import EMAIL_RECEIVER, SMTP_SERVER, SMTP_PORT

class AlertNotifier:
    def __init__(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def send_alert(self, subject, message):
        """Send email alert to your personal email"""
        msg = MIMEText(message, 'html')
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = EMAIL_RECEIVER
        
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
            return False