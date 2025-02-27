import streamlit as st
import pandas as pd
from data_processing import load_data, preprocess_data, get_top_features
from models import train_models, evaluate_models, plot_confusion_matrix, get_classification_metrics
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

# Добавляем фильтр по столбцам
st.write("Фильтр данных")
selected_columns = st.multiselect("Выберите столбцы для отображения", df.columns, default=df.columns)
st.dataframe(df[selected_columns])


# Обработка данных
X_train, X_test, y_train, y_test, top2, top3 = preprocess_data(df)

# Построение графиков
st.write("### Матрица корреляции")
st.plotly_chart(plot_correlation_matrix(df))

st.write("### 3D Scatter Plot")
st.plotly_chart(plot_3d_scatter(df, top3))

# Обучаем модели только на топ-2 признаках
st.sidebar.subheader("Настройки моделей")

# Гиперпараметры
logistic_c = st.sidebar.slider("Logistic Regression (C)", 0.01, 10.0, 1.0)
knn_neighbors = st.sidebar.slider("KNN (количество соседей)", 1, 15, 3)
tree_depth = st.sidebar.slider("Decision Tree (глубина)", 1, 20, 5)

custom_params = {
    "Logistic Regression": {"C": logistic_c, "max_iter": 1000},
    "KNN": {"n_neighbors": knn_neighbors, "weights": "distance"},
    "Decision Tree": {"max_depth": tree_depth, "criterion": "entropy"}
}

st.sidebar.subheader("Выберите модели для обучения")
use_logistic = st.sidebar.checkbox("Logistic Regression", value=True)
use_knn = st.sidebar.checkbox("KNN", value=True)
use_tree = st.sidebar.checkbox("Decision Tree", value=True)

selected_models = {}
if use_logistic:
    selected_models["Logistic Regression"] = custom_params["Logistic Regression"]
if use_knn:
    selected_models["KNN"] = custom_params["KNN"]
if use_tree:
    selected_models["Decision Tree"] = custom_params["Decision Tree"]

st.write("### Обучение моделей")
models = train_models(X_train, y_train, top2, hyperparams=selected_models)


# Отображение границ решений
st.write("### Границы решений для топ-2 признаков")
for model_name, model in models.items():
    st.pyplot(plot_decision_boundaries(model, X_train[top2].values, y_train.values, f"Границы решений для {model_name}", top2[0], top2[1]))

# Оценка моделей
st.write("### ROC-кривые")
roc_auc_df, fprs, tprs, aucs = evaluate_models(models, X_test[top2].values, y_test.values, X_train[top2].values, y_train.values)
selected_roc_models = st.multiselect("Выберите модели для отображения ROC-кривых", list(models.keys()), default=list(models.keys()))

filtered_fprs = {name: fprs[name] for name in selected_roc_models}
filtered_tprs = {name: tprs[name] for name in selected_roc_models}
filtered_aucs = {name: aucs[name] for name in selected_roc_models}

st.pyplot(plot_roc_curve(filtered_fprs, filtered_tprs, filtered_aucs))

# Метрики моделей
st.write("### Оценка качества классификации")
st.dataframe(roc_auc_df.round(3))

st.write("### Метрики классификации")

selected_eval_model = st.selectbox("Выберите модель для оценки", list(models.keys()))

if selected_eval_model:
    model = models[selected_eval_model]

    # Делаем предсказание
    y_pred = model.predict(X_test[top2])

     # Выводим Precision, Recall, F1-score
    st.write(f"### Precision, Recall, F1-score - {selected_eval_model}")
    class_report = get_classification_metrics(y_test, y_pred)
    st.table(class_report)

    # Выводим Confusion Matrix
    st.write(f"### Confusion Matrix - {selected_eval_model}")
    st.pyplot(plot_confusion_matrix(y_test, y_pred, selected_eval_model))

   

    