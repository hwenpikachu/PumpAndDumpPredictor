import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class SimplePumpDumpDetector:
    """
    Simple pump and dump detector.
    - Trains on past 5 days (excluding last 24 hours)
    - Detects anomalies in the last 24 hours
    """
    
    def __init__(self, product_id='SAPIEN-USD'):
        self.product_id = product_id
        self.base_url = 'https://api.exchange.coinbase.com'
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=0.01,
            random_state=42,
            n_estimators=100
        )

    # Fetch historical candle data from Coinbase API
    def fetch_candles(self, granularity=300, lookback_hours=144):
        candles_per_request = 300
        total_candles_needed = (lookback_hours * 3600) // granularity
        all_data = []
        end_time = datetime.utcnow()
        num_requests = min(int(np.ceil(total_candles_needed / candles_per_request)), 20)
        
        print(f"Fetching data in {num_requests} batches...")
        
        for i in range(num_requests):
            batch_end = end_time - timedelta(hours=i * (candles_per_request * granularity / 3600))
            batch_start = batch_end - timedelta(hours=(candles_per_request * granularity / 3600))
            
            url = f'{self.base_url}/products/{self.product_id}/candles'
            params = {
                'start': int(batch_start.timestamp()),
                'end': int(batch_end.timestamp()),
                'granularity': granularity
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    all_data.extend(data)
                    
                if i < num_requests - 1:
                    import time
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error fetching batch {i+1}: {e}")
                if i == 0:
                    return None
                break
        
        if not all_data:
            return None
            
        df = pd.DataFrame(all_data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop_duplicates(subset=['time'])
        
        for col in ['low', 'high', 'open', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        return df
    
    # Create simple data features for our detection
    def create_features(self, df):
        features = pd.DataFrame(index=df.index)
        
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['returns_6'] = df['close'].pct_change(periods=6)
        df['returns_12'] = df['close'].pct_change(periods=12)
        
        # Volume
        df['volume_ma_24'] = df['volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_24'] + 1e-10)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=12).std()
        
        # Price range
        df['price_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        
        # Features for model
        features['returns'] = df['returns']
        features['returns_6'] = df['returns_6']
        features['returns_12'] = df['returns_12']
        features['volume_ratio'] = df['volume_ratio']
        features['volatility'] = df['volatility']
        features['price_range'] = df['price_range']
        
        return features, df
    
    def train_and_detect(self):
        """
        Main method: 
        1. Fetch 6 days of data (144 hours)
        2. Train on days 1-5 (hours 24-144)
        3. Detect anomalies in last 24 hours
        """
        print(f"\n{'='*70}")
        print(f"PUMP & DUMP DETECTOR - {self.product_id}")
        print(f"{'='*70}\n")
        
        # Fetch 6 days of data
        print("Step 1: Fetching last 6 days of data...")
        df = self.fetch_candles(granularity=300, lookback_hours=144)
        
        if df is None or len(df) == 0:
            print("Error: Could not fetch data")
            return None
        
        print(f"✓ Fetched {len(df)} candles\n")
        
        # Split data: last 24 hours vs older data
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        df_train = df[df['time'] < cutoff_time].copy()
        df_test = df[df['time'] >= cutoff_time].copy()
        
        print(f"Step 2: Splitting data...")
        print(f"  Training data: {len(df_train)} candles (days 2-6)")
        print(f"  Test data: {len(df_test)} candles (last 24 hours)\n")
        
        if len(df_train) < 50:
            print("Error: Not enough training data")
            return None
        
        if len(df_test) == 0:
            print("Error: No recent data to analyze")
            return None
        
        # Create features
        print("Step 3: Creating features...")
        features_train, df_train = self.create_features(df_train)
        features_test, df_test = self.create_features(df_test)
        
        # Remove NaN
        features_train = features_train.dropna()
        features_test = features_test.dropna()
        
        if len(features_train) < 50:
            print("Error: Not enough valid training features")
            return None
        
        print(f"  Training features: {len(features_train)} samples")
        print(f"  Test features: {len(features_test)} samples\n")
        
        # Train model
        print("Step 4: Training anomaly detection model...")
        X_train = self.scaler.fit_transform(features_train)
        self.model.fit(X_train)
        print("✓ Model trained\n")
        
        # Detect anomalies in last 24 hours
        print("Step 5: Detecting anomalies in last 24 hours...")
        X_test = self.scaler.transform(features_test)
        predictions = self.model.predict(X_test)
        anomaly_scores = self.model.score_samples(X_test)
        
        # Create results
        results = features_test.copy()
        results['anomaly'] = predictions == -1
        results['anomaly_score'] = anomaly_scores
        results['time'] = df_test.loc[features_test.index, 'time']
        results['price'] = df_test.loc[features_test.index, 'close']
        results['volume'] = df_test.loc[features_test.index, 'volume']
        results['volume_ratio'] = features_test['volume_ratio']
        
        # Count anomalies
        num_anomalies = (predictions == -1).sum()
        anomaly_pct = (num_anomalies / len(predictions)) * 100
        
        print(f"✓ Found {num_anomalies} anomalies ({anomaly_pct:.1f}% of last 24 hours)\n")
        
        # Calculate statistics
        price_change_24h = ((df_test['close'].iloc[-1] / df_test['close'].iloc[0]) - 1) * 100
        max_volume_surge = results['volume_ratio'].max()
        avg_volume_surge = results['volume_ratio'].mean()
        
        # Determine if pump/dump detected
        is_pump_dump = False
        risk_level = "LOW"


        # If less than 5% daily change, never call it manipulation
        if abs(price_change_24h) < 5:
            is_pump_dump = False
            risk_level = "LOW"

        else:
            # anomaly-based
            if num_anomalies > len(predictions) * 0.30:
                is_pump_dump = True
                risk_level = "CRITICAL"
            elif num_anomalies > len(predictions) * 0.15:
                is_pump_dump = True
                risk_level = "HIGH"
            elif num_anomalies > len(predictions) * 0.10:
                risk_level = "MEDIUM"

            # price movement rules
            if abs(price_change_24h) > 50:
                is_pump_dump = True
                risk_level = "CRITICAL"
            elif abs(price_change_24h) > 20:
                risk_level = max(risk_level, "HIGH")
        
        # Print report
        self.print_report(results, price_change_24h, max_volume_surge, 
                         avg_volume_surge, is_pump_dump, risk_level, df_test)
        
        return {
            'results': results,
            'is_pump_dump': is_pump_dump,
            'risk_level': risk_level,
            'price_change_24h': price_change_24h,
            'num_anomalies': num_anomalies,
            'anomaly_pct': anomaly_pct
        }
    
    def print_report(self, results, price_change_24h, max_volume_surge, 
                    avg_volume_surge, is_pump_dump, risk_level, df_test):
        """Print detection report."""
        print(f"{'='*70}")
        print(f"DETECTION REPORT")
        print(f"{'='*70}\n")
        
        print(f"Product: {self.product_id}")
        print(f"Current Price: ${df_test['close'].iloc[-1]:.6f}")
        print(f"24h Price Change: {price_change_24h:+.2f}%\n")
        
        print(f"RISK LEVEL: {risk_level}")
        print(f"Pump/Dump Detected: {'YES' if is_pump_dump else 'NO ✓'}\n")
        
        print(f"Anomaly Statistics (Last 24 Hours):")
        print(f"  Total anomalies: {(results['anomaly']).sum()} / {len(results)}")
        print(f"  Anomaly rate: {((results['anomaly']).sum() / len(results) * 100):.1f}%")
        print(f"  Max volume surge: {max_volume_surge:.2f}x")
        print(f"  Avg volume surge: {avg_volume_surge:.2f}x\n")
        
        # Show most anomalous periods
        anomalies = results[results['anomaly']].copy()
        if len(anomalies) > 0:
            print(f"Top 5 Most Anomalous Time Periods:")
            anomalies_sorted = anomalies.sort_values('anomaly_score')
            for idx, row in anomalies_sorted.head(5).iterrows():
                print(f"  {row['time']} - Price: ${row['price']:.6f}, "
                      f"Volume: {row['volume_ratio']:.2f}x, Score: {row['anomaly_score']:.3f}")
        else:
            print("No significant anomalies detected in last 24 hours")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print("Pump & Dump Detector")
    print("Trains on past 5 days, detects in last 24 hours\n")
    
    # Initialize detector
    detector = SimplePumpDumpDetector(product_id='SAPIEN-USD')
    
    # Run detection
    result = detector.train_and_detect()
    
    if result:
        print("\n")
        print("  - Change product_id to test other coins: 'ETH-USD', 'DOGE-USD', etc.")
        print("  - Run this every few hours to monitor for new pump/dump activity")
        print("  - Anomaly rate >30% strongly indicates manipulation")