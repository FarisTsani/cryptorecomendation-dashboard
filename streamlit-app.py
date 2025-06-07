import streamlit as st
import pandas as pd
import numpy as np

from preprocessing import load_data, merge_and_clean_data, feature_engineering, encode_coin_column
from model_training import train_random_forest, train_svm, train_xgboost
from predictor import predict_next_7_days, generate_trading_recommendation
from visualization import calculate_technical_indicators

import plotly.graph_objs as go
import plotly.subplots as sp

# ====== SETUP STREAMLIT UI ======
st.set_page_config(page_title="Crypto Predictor", layout="wide")
st.title("🚀 Crypto Price Prediction & Analysis App")

# ==== LOAD DATA ====
coin_file = pd.read_excel("top_50_coin.xlsx")
tvl_file = pd.read_excel("tVL.xlsx")

if 'data' not in st.session_state:
    df, df_tvl = load_data(coin_file, tvl_file)
    merged_df = merge_and_clean_data(df, df_tvl)
    data = feature_engineering(merged_df)
    data = encode_coin_column(data)

    coin_names = merged_df['coin'].unique().tolist()
    coin_code_to_name = {i: name for i, name in enumerate(coin_names)}

    st.session_state['data'] = data
    st.session_state['coin_code_to_name'] = coin_code_to_name
    st.session_state['charts'] = {
        name: calculate_technical_indicators(data[data['coin'] == code].copy())
        for code, name in coin_code_to_name.items()
    }

# ==== CHART VISUALIZATION (selalu tampil) ====
st.subheader("📈 Technical Chart Visualization")
selected_coin = st.selectbox(
    "Pilih koin untuk dilihat chart-nya:",
    list(st.session_state['charts'].keys())
)

chart_df = st.session_state['charts'].get(selected_coin)
if chart_df is not None and not chart_df.empty:
    chart_df['date'] = pd.to_datetime(chart_df['date'])

    fig = sp.make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Harga & Moving Averages", "RSI", "MACD")
    )

    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['close'], mode='lines', name='Price', line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['SMA5'], mode='lines', name='5MA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['SMA10'], mode='lines', name='10MA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['SMA20'], mode='lines', name='20MA', line=dict(color='green')), row=1, col=1)

    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['RSI14'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)

    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['MACD'], mode='lines', name='MACD', line=dict(color='magenta')), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_df['date'], y=chart_df['MACD_signal'], mode='lines', name='Signal', line=dict(color='gold')), row=3, col=1)
    fig.add_trace(go.Bar(x=chart_df['date'], y=chart_df['MACD_histogram'], name='Histogram', marker_color='skyblue', opacity=0.5), row=3, col=1)

    fig.update_layout(
        height=850,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Data chart untuk koin ini tidak tersedia.")

# ====== PREDICTION BUTTON ======
st.subheader("🧠 Jalankan Prediksi Harga Koin")
if st.button("🔮 Predict"):
    with st.spinner("Melatih model dan memprediksi..."):
        features = ['coin', 'close', 'SMA7', 'SMA14', 'RSI14', 'volume', 'tvl']
        target = 'target'

        data = st.session_state['data']
        rf_models = train_random_forest(data, features, target)
        svm_models = train_svm(data, features, target)
        xgb_models = train_xgboost(data, features, target)

        models = {
            'Random Forest': rf_models,
            'SVM': svm_models,
            'XGBoost': xgb_models
        }

        prediction_results = []

        for coin_code in data['coin'].unique():
            coin_data = data[data['coin'] == coin_code]
            current_price = coin_data['close'].iloc[-1]
            coin_name = st.session_state['coin_code_to_name'].get(coin_code, f"Unknown-{coin_code}")

            try:
                rf_f1 = float(models['Random Forest'][coin_code]['classification_report'].split('\n')[2].split()[3])
                svm_f1 = float(models['SVM'][coin_code]['classification_report'].split('\n')[2].split()[3])
                xgb_f1 = float(models['XGBoost'][coin_code]['classification_report'].split('\n')[2].split()[3])

                best_model = 'Random Forest' if rf_f1 >= max(svm_f1, xgb_f1) else ('SVM' if svm_f1 >= xgb_f1 else 'XGBoost')

                predictions = predict_next_7_days(coin_code, best_model, models, coin_data, features)
                recommendation, reason, change_pct = generate_trading_recommendation(predictions, current_price)

                result = {
                    'Koin': coin_name,
                    'Best Model': best_model,
                    'Current Price': f"${current_price:.6f}",
                    **{f"Day {i+1}": f"${p['predicted_price']:.6f}" for i, p in enumerate(predictions)},
                    'Total Change %': f"{change_pct:+.1f}%",
                    'Recommendation': recommendation,
                    'Reason': reason
                }
                prediction_results.append(result)

            except Exception as e:
                st.warning(f"Prediction failed for coin {coin_name}: {e}")

        final_df = pd.DataFrame(prediction_results)
        st.session_state['final_df'] = final_df

# ====== DISPLAY PREDICTION RESULT ======
if 'final_df' in st.session_state:
    st.subheader("📊 Prediction Results")
    st.dataframe(st.session_state['final_df'])

    top_5 = st.session_state['final_df'][st.session_state['final_df']['Recommendation'] == 'LONG']\
        .sort_values('Total Change %', ascending=False).head(5)
    st.subheader("🚀 Top 5 Recommended Coins")
    st.dataframe(top_5[['Koin', 'Total Change %', 'Reason']])

    st.download_button(
        label="📥 Download Full Prediction Results",
        data=st.session_state['final_df'].to_csv(index=False).encode('utf-8'),
        file_name="crypto_predictions.csv",
        mime="text/csv"
    )
