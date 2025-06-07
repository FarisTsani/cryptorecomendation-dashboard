# predictor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def calculate_technical_indicators(df):
    df['SMA7'] = df['close'].rolling(window=7).mean()
    df['SMA14'] = df['close'].rolling(window=14).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    return df

def predict_next_7_days(coin_code, best_model_name, model_dicts, current_data, features):
    model = model_dicts[best_model_name][coin_code]['model']
    scaler = StandardScaler()
    scaler.fit(current_data[features])

    predictions = []
    temp_data = current_data.copy()

    for day in range(7):
        last_row = temp_data.iloc[-1:].copy()
        feature_row = last_row[features].values.reshape(1, -1)
        feature_row_scaled = scaler.transform(feature_row)

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_row_scaled)[0]
            prediction = model.predict(feature_row_scaled)[0]
        else:
            prediction = model.predict(feature_row_scaled)[0]
            proba = [0.5, 0.5]

        current_price = last_row['close'].iloc[0]
        confidence = max(proba)
        price_change = current_price * (0.01 + 0.04 * confidence)
        predicted_price = current_price + price_change if prediction == 1 else current_price - price_change

        predictions.append({
            'day': day + 1,
            'predicted_price': predicted_price,
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': confidence
        })

        new_row = last_row.copy()
        new_row['close'] = predicted_price
        temp_data = pd.concat([temp_data, new_row], ignore_index=True)
        temp_data = calculate_technical_indicators(temp_data)
        temp_data.fillna(method='ffill', inplace=True)

    return predictions

def generate_trading_recommendation(predictions, current_price):
    day7_price = predictions[-1]['predicted_price']
    total_change_pct = ((day7_price - current_price) / current_price) * 100
    up_days = sum(1 for p in predictions if p['direction'] == 'UP')
    down_days = sum(1 for p in predictions if p['direction'] == 'DOWN')
    avg_confidence = np.mean([p['confidence'] for p in predictions])

    if total_change_pct > 3 and up_days >= 5 and avg_confidence > 0.6:
        recommendation = "LONG"
    elif total_change_pct < -3 and down_days >= 5 and avg_confidence > 0.6:
        recommendation = "SHORT"
    elif abs(total_change_pct) <= 2 or avg_confidence < 0.55:
        recommendation = "HOLD"
    elif total_change_pct > 0 and up_days > down_days:
        recommendation = "LONG"
    else:
        recommendation = "SHORT"

    reason = f"Change: {total_change_pct:.1f}%, Up days: {up_days}, Confidence: {avg_confidence:.2f}"
    return recommendation, reason, total_change_pct
