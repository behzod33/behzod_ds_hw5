import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from mlxtend.plotting import plot_decision_regions

# --- Основные настройки ---
st.set_page_config(page_title="Streamlit Проект", layout="wide")

# --- Функция загрузки данных ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

# --- Функция предобработки данных ---
@st.cache_data
def preprocess_data(df):
    X = df.drop(columns="target")
    y = df["target"]

    # Стандартизация
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# --- Функция для отрисовки границ решений ---
def plot_decision_boundaries(model, X, y, title):
    """Построение границ решений."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_decision_regions(X, y, clf=model, legend=2, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    st.pyplot(fig)

# --- Функция для обучения моделей только на 2 признаках ---
def train_models(X_train, y_train):
    """Обучение моделей на 2 признаках."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=5)
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

# --- Функция для отображения ROC-кривой ---
def plot_roc_curve(y_test, model_probs, model_names):
    """Построение ROC-кривой."""
    plt.figure(figsize=(10, 7))
    for name, y_pred in zip(model_names, model_probs):
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые для моделей")
    plt.legend(loc="lower right")
    st.pyplot(plt)

# --- Заголовок приложения ---
st.title("🏆 Streamlit-проект для классификации данных")

# --- Боковая панель ---
st.sidebar.header("⚙️ Настройки")
uploaded_file = st.sidebar.file_uploader("📂 Загрузите CSV файл с данными", type=["csv"])

# --- Основной контент ---
if uploaded_file:
    # Загрузка данных
    df = load_data(uploaded_file)
    st.write("## 🔍 Загруженный DataFrame")
    st.dataframe(df)

    # Проверка наличия целевой переменной
    if "target" not in df.columns:
        st.error("🚫 В DataFrame нет колонки 'target'!")
        st.stop()

    # Предобработка данных
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # **Выбираем топ-2 признака**
    top2_features = X_train.corr().abs().sum().sort_values(ascending=False).index[:2]

    st.write(f"### 🏆 Топ-2 признака: {', '.join(top2_features)}")

    # Используем только эти признаки для обучения
    X_train_2d = X_train[top2_features].values
    X_test_2d = X_test[top2_features].values

    # --- Корреляционная матрица ---
    st.write("## 📊 Матрица корреляции")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --- 3D Scatter Plot ---
    st.write("## 🌌 3D Scatter Plot")
    if len(df.columns) >= 4:
        scatter_fig = px.scatter_3d(
            df,
            x=df.columns[0],
            y=df.columns[1],
            z=df.columns[2],
            color=df["target"].astype(str),
            title="3D Scatter Plot для бинарной классификации"
        )
        st.plotly_chart(scatter_fig)
    else:
        st.warning("⚠️ Недостаточно признаков для 3D визуализации.")

    # --- Обучение моделей ---
    st.write("## 🤖 Обучение моделей на топ-2 признаках")
    models = train_models(X_train_2d, y_train)

    # --- Границы решений ---
    st.write("## 🟢 Границы решений для моделей")

    for name, model in models.items():
        st.write(f"### {name}")
        plot_decision_boundaries(model, X_train_2d, y_train.values, f"Границы решений для {name}")

    # --- Оценка моделей ---
    st.write("## 📈 Оценка моделей")
    model_probs = []
    model_names = []

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test_2d)[:, 1]
        y_pred = model.predict(X_test_2d)

        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        st.write(f"**{name}**")
        st.write(f"- Точность: {accuracy:.2f}")
        st.write(f"- AUC: {auc_score:.2f}")
        st.write("")

        model_probs.append(y_pred_proba)
        model_names.append(name)

    # --- ROC-кривая ---
    st.write("## 🚀 ROC-кривые для моделей")
    plot_roc_curve(y_test, model_probs, model_names)

else:
    st.info("📥 Пожалуйста, загрузите CSV файл с данными для начала работы.")
