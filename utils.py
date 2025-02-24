import streamlit as st

def display_model_metrics(metrics_df):
    """Отображение метрик моделей."""
    st.write("### 📊 Оценка качества классификации")
    st.dataframe(metrics_df.round(3))
