import streamlit as st
import pandas as pd
from data_processing import load_data, preprocess_data, get_top_features
from models import train_models, evaluate_models
from plots import plot_correlation_matrix, plot_3d_scatter, plot_decision_boundaries, plot_roc_curve

# Загрузка данных
st.title("Домашнее задание 5: Классификация")

st.sidebar.header("Настройки")
path = st.sidebar.text_input("Укажите путь к CSV-файлу", "data/glass.data")
uploaded_file = st.sidebar.file_uploader("Загрузите CSV-файл", type=["csv", "data"])

# Загрузка данных
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data(path) 

if df is None:
    st.error("Ошибка загрузки данных. Проверьте путь к файлу.")
    st.stop()

# Отображение данных
st.write("### Исходный DataFrame")
st.dataframe(df)

# Обработка данных
X_train, X_test, y_train, y_test, top2, top3 = preprocess_data(df)

# Построение графиков
st.write("### Матрица корреляции")
st.pyplot(plot_correlation_matrix(df))

st.write("### 3D Scatter Plot")
st.plotly_chart(plot_3d_scatter(df, top3))

# Обучаем модели только на топ-2 признаках
st.write("### Обучение моделей")
models = train_models(X_train, y_train, top2)


# Отображение границ решений
st.write("### Границы решений для топ-2 признаков")
for model_name, model in models.items():
    st.pyplot(plot_decision_boundaries(model, X_train[top2].values, y_train.values, f"Границы решений для {model_name}", top2[0], top2[1]))

# Оценка моделей
st.write("### ROC-кривые")
roc_auc_df, fprs, tprs, aucs = evaluate_models(models, X_test[top2].values, y_test.values, X_train[top2].values, y_train.values)
st.pyplot(plot_roc_curve(fprs, tprs, aucs))

# Метрики моделей
st.write("### Оценка качества классификации")
st.dataframe(roc_auc_df.round(3))

