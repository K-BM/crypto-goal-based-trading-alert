import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

class BTCPredictor:
    def __init__(self, threshold_pct=np.arange(0.1, 3, 0.1), direction='increase'):
        self.threshold_pct = threshold_pct
        self.direction = direction
        if direction not in ['increase', 'decrease']:
            raise ValueError("Direction must be either 'increase' or 'decrease'")
    
    @staticmethod
    def prepare_features(daily_df):
        """Select final features and split data"""
        # Exclude raw prices and forward-looking data
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
        features = [col for col in daily_df.columns if col not in exclude]
        
        X = daily_df[features]
        y = daily_df['target']
        
        # Time-based split
        tscv = TimeSeriesSplit(n_splits=5)
        splits = []
        for train_idx, test_idx in tscv.split(X):
            splits.append({
                'X_train': X.iloc[train_idx],
                'X_test': X.iloc[test_idx],
                'y_train': y.iloc[train_idx],
                'y_test': y.iloc[test_idx]
            })
        
        return splits

    @staticmethod
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        """Enhanced evaluation with threshold optimization"""
        # Get predictions
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold on TRAIN set
        precision, recall, thresholds = precision_recall_curve(y_train, y_proba_train)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Evaluate both sets at optimal threshold
        y_pred_train = (y_proba_train >= optimal_threshold).astype(int)
        y_pred_test = (y_proba_test >= optimal_threshold).astype(int)
        
        # Calculate metrics
        def get_metrics(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
            specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
            return accuracy, precision, recall, f1, tn, fp, fn, tp, sensitivity, specificity
        
        train_metrics = get_metrics(y_train, y_pred_train)
        test_metrics = get_metrics(y_test, y_pred_test)
        
        # Calculate overfitting gaps
        accuracy_gap = train_metrics[0] - test_metrics[0]
        f1_gap = train_metrics[3] - test_metrics[3]
        
        return {
            'y_proba_test': y_proba_test,
            'optimal_threshold': optimal_threshold,
            'metrics': {
                'train': {
                    'accuracy': train_metrics[0],
                    'precision': train_metrics[1],
                    'recall': train_metrics[2],
                    'f1': train_metrics[3],
                    'tn': train_metrics[4],
                    'fp': train_metrics[5],
                    'fn': train_metrics[6],
                    'tp': train_metrics[7],
                    'sensitivity': train_metrics[8],
                    'specificity': train_metrics[9]
                },
                'test': {
                    'accuracy': test_metrics[0],
                    'precision': test_metrics[1],
                    'recall': test_metrics[2],
                    'f1': test_metrics[3],
                    'tn': test_metrics[4],
                    'fp': test_metrics[5],
                    'fn': test_metrics[6],
                    'tp': test_metrics[7],
                    'sensitivity': test_metrics[8],
                    'specificity': test_metrics[9]
                },
                'gaps': {
                    'accuracy': accuracy_gap,
                    'f1': f1_gap
                }
            }
        }

    def evaluate_multiple_thresholds(self, daily_with_indicators):
        """Evaluate model performance across multiple threshold percentages with consistent test set"""
        results = []
        
        # Create base dataset (without target)
        daily_data_full = daily_with_indicators.copy()
        feature_cols = [c for c in daily_data_full.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        X_full = daily_data_full[feature_cols]
        
        # Single time-based split for all thresholds
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X_full))
        train_idx, test_idx = splits[-1]  # Use most recent split
        
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_all, y_test_all = daily_data_full.iloc[train_idx], daily_data_full.iloc[test_idx]
        
        # Get latest observation for prediction check
        last_data = X_test.iloc[[-1]]
        last_timestamp = last_data.index[0]
        last_open = daily_data_full.loc[last_timestamp, 'open']
        last_high = daily_data_full.loc[last_timestamp, 'high']
        last_low = daily_data_full.loc[last_timestamp, 'low']
        
        for threshold in self.threshold_pct:
            # Create targets based on direction
            if self.direction == 'increase':
                y_train = (y_train_all['high'] >= y_train_all['open'] * (1 + threshold / 100)).astype(int)
                y_test = (y_test_all['high'] >= y_test_all['open'] * (1 + threshold / 100)).astype(int)
                actual_move = int(last_high >= last_open * (1 + threshold / 100))
                target_price = last_open * (1 + threshold / 100)
            else:  # decrease
                y_train = (y_train_all['low'] <= y_train_all['open'] * (1 - threshold / 100)).astype(int)
                y_test = (y_test_all['low'] <= y_test_all['open'] * (1 - threshold / 100)).astype(int)
                actual_move = int(last_low <= last_open * (1 - threshold / 100))
                target_price = last_open * (1 - threshold / 100)
            
            # Handle imbalance with scale_pos_weight
            scale_pos_weight = len(y_train[y_train==0]) / max(1, len(y_train[y_train==1]))
            
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=4,
                learning_rate=0.03,
                n_estimators=150,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=20
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            
            eval_results = self.evaluate_model(model, X_train, y_train, X_test, y_test)
            
            last_proba = model.predict_proba(last_data)[0, 1]
            last_pred = int(last_proba >= eval_results['optimal_threshold'])
            
            results.append({
                'Threshold (%)': threshold,
                'Direction': self.direction.capitalize(),
                'Optimal_Threshold': eval_results['optimal_threshold'],
                
                # Test metrics
                'Test_Accuracy': eval_results['metrics']['test']['accuracy'],
                'Test_Precision': eval_results['metrics']['test']['precision'],
                'Test_Recall': eval_results['metrics']['test']['recall'],
                'Test_F1': eval_results['metrics']['test']['f1'],
                'Test_Sensitivity': eval_results['metrics']['test']['sensitivity'],
                'Test_Specificity': eval_results['metrics']['test']['specificity'],
                
                # Train metrics
                'Train_Accuracy': eval_results['metrics']['train']['accuracy'],
                'Train_Precision': eval_results['metrics']['train']['precision'],
                'Train_Recall': eval_results['metrics']['train']['recall'],
                'Train_F1': eval_results['metrics']['train']['f1'],
                'Train_Sensitivity': eval_results['metrics']['train']['sensitivity'],
                'Train_Specificity': eval_results['metrics']['train']['specificity'],
                
                # Overfitting gaps
                'Accuracy_Gap': eval_results['metrics']['gaps']['accuracy'],
                'F1_Gap': eval_results['metrics']['gaps']['f1'],
                
                # Prediction info
                'Latest_Proba': last_proba,
                'Latest_Pred': last_pred,
                'Actual_Move': actual_move,
                'Open_Price': last_open,
                'Target_Price': target_price,
                'Timestamp': last_timestamp.strftime('%Y-%m-%d %H:%M'),
                'Prediction_Time_UTC': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return pd.DataFrame(results)