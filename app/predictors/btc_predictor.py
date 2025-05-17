import pandas as pd
import numpy as np
import xgboost as xgb
from app.data.processors import DataProcessor  
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

class BTCPredictor:
    def __init__(self, threshold_pct=np.arange(0.1, 3, 0.05)):
        self.threshold_pct = threshold_pct
    
    def prepare_features(self, daily_df):
        """Select final features and split data"""
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
        features = [col for col in daily_df.columns if col not in exclude]
        X = daily_df[features]
        y = daily_df['target']
        return X, y
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Model evaluation with threshold optimization"""
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_train, y_proba_train)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        
        # Evaluate
        y_pred_test = (y_proba_test >= optimal_threshold).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test),
            'recall': recall_score(y_test, y_pred_test),
            'f1': f1_score(y_test, y_pred_test)
        }
        
        return metrics, optimal_threshold
    
    def evaluate_multiple_thresholds(self, hourly_with_indicators):
        """Evaluate across multiple thresholds"""
        results = []
        
        # Create base dataset
        daily_data_full = DataProcessor.create_daily_dataset(hourly_with_indicators, 1.0)
        X_full = daily_data_full[[c for c in daily_data_full.columns 
                               if c not in ['open','high','low','close','volume','target']]]
        
        # Time-based split
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(X_full):
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train_all, y_test_all = daily_data_full.iloc[train_idx], daily_data_full.iloc[test_idx]
        
        # Latest observation
        last_data = X_test.iloc[[-1]]
        last_timestamp = last_data.index[0]
        last_open = daily_data_full.loc[last_timestamp, 'open']
        last_high = daily_data_full.loc[last_timestamp, 'high']
        
        for threshold_pct in self.threshold_pct:
            # Create targets
            y_train = (y_train_all['high'] >= y_train_all['open']*(1 + threshold_pct/100)).astype(int)
            y_test = (y_test_all['high'] >= y_test_all['open']*(1 + threshold_pct/100)).astype(int)
            
            # Train model
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=4,
                learning_rate=0.03,
                n_estimators=150,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                early_stopping_rounds=20
            )
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
            
            # Evaluate
            metrics, optimal_threshold = self.evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Latest prediction
            last_proba = model.predict_proba(last_data)[0,1]
            last_pred = int(last_proba >= optimal_threshold)
            actual_move = int(last_high >= last_open*(1 + threshold_pct/100))
            
            results.append({
                'Threshold (%)': threshold_pct,
                **metrics,
                'Latest_Prediction': last_pred,
                'Actual_Move': actual_move,
                'Open_Price': last_open,
                'Target_Price': last_open * (1 + threshold_pct/100)
            })
        
        return pd.DataFrame(results)