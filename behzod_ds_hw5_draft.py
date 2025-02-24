import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

from mlxtend.plotting import plot_decision_regions

# Часть с переобработкой данных сделана заранее чтобы не нагружать основной файл

@st.cache_data
def get_df(path="data/data_hw5.csv"):
    try:
        return pd.read_csv(path)
    except:
        st.write("Не правилный путь у csv файлу!!!")
        

def plot_dr(model, X_train, y_train, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_train, y_train, clf=model, legend=2, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig

# Построение графика
def plot_roc_curve():
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    
    ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.2f})")
    ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.2f})")
    ax.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="dotted")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC-кривые для разных моделей")
    ax.legend(loc="lower right")
    ax.grid(True)

    return fig

df = get_df()

cor_mat = df.corr().round(2)[:-1]
top3 = cor_mat.abs().sort_values(by="target", ascending=False)[:3].index.tolist()
top2 = cor_mat.abs().sort_values(by="target", ascending=False)[:2].index.tolist()

cor_mat_plt, cor_mat_ax = plt.subplots(figsize=(8, 6), dpi=200)
sns.heatmap(cor_mat, annot=True, cmap="coolwarm")
cor_mat_ax.set_title("Матрица корреляции")

color_map = {"0": "orange", "1": "blue"}

scatt_3d = px.scatter_3d(
    df, 
    x=top3[2], 
    y=top3[1], 
    z=top3[0], 
    color=df["target"].astype(str), 
    color_discrete_map=color_map,
    title="3D Scatter Plot для Бинарной классификации",
    labels={
        "color": "Тип стекла"
    }
)

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

scaler = StandardScaler()

columns = X_train.columns
X_train[columns] = scaler.fit_transform(X_train[columns])
X_test[columns] = scaler.transform(X_test[columns])


X_train_final = X_train[top2].values
X_test_final = X_test[top2].values

y_train_final = y_train.values
y_test_final = y_test.values

model_knn = KNeighborsClassifier(n_neighbors=3)
model_lr = LogisticRegression(max_iter=565)
model_dt = DecisionTreeClassifier(max_depth=5)

model_knn.fit(X_train_final, y_train_final)
model_lr.fit(X_train_final, y_train_final)
model_dt.fit(X_train_final, y_train_final);


# ---- TRAIN ROC AUC -----

y_train_lr = model_lr.predict_proba(X_train_final)[:, 1]
y_train_knn = model_knn.predict_proba(X_train_final)[:, 1]
y_train_dt = model_dt.predict_proba(X_train_final)[:, 1]

roc_train_lr = roc_auc_score(y_train_final, y_train_lr)
roc_train_knn = roc_auc_score(y_train_final, y_train_knn)
roc_train_dt = roc_auc_score(y_train_final, y_train_dt)

# ---- TEST ROC AUC -----

y_test_lr = model_lr.predict_proba(X_test_final)[:, 1]
y_test_knn = model_knn.predict_proba(X_test_final)[:, 1]
y_test_dt = model_dt.predict_proba(X_test_final)[:, 1]

roc_test_lr = roc_auc_score(y_test_final, y_test_lr)
roc_test_knn = roc_auc_score(y_test_final, y_test_knn)
roc_test_dt = roc_auc_score(y_test_final, y_test_dt)


roc_auc_df = pd.DataFrame({
    "Модель": ["Logistic Regression", "KNN", "Decision Tree"],
    "Train AUC": [roc_train_lr, roc_train_knn, roc_train_dt],
    "Test AUC": [roc_test_lr, roc_test_knn, roc_test_dt],
    "Разница AUC": [
        abs(roc_train_lr - roc_test_lr),
        abs(roc_train_knn - roc_test_knn),
        abs(roc_train_dt - roc_test_dt)
    ]
})

fpr_lr, tpr_lr, _ = roc_curve(y_test_final, y_test_lr)
fpr_knn, tpr_knn, _ = roc_curve(y_test_final, y_test_knn)
fpr_dt, tpr_dt, _ = roc_curve(y_test_final, y_test_dt)

auc_lr = auc(fpr_lr, tpr_lr)
auc_knn = auc(fpr_knn, tpr_knn)
auc_dt = auc(fpr_dt, tpr_dt)



st.title("Домашнее задание 5")

st.write("Датафрейм")

st.dataframe(data=df)


st.write(f"Топ 2: {", ".join(top2)}")
st.write(f"Топ 3: {", ".join(top3)}")

st.pyplot(cor_mat_plt)

st.plotly_chart(scatt_3d)

st.write("X_train после стандартизации")
st.dataframe(X_train.describe().round(3))

title_lr = "Границы решений для LR"
title_knn="Границы решений для KNN"
title_dt="Границы решений для DT"

xlabel=top2[0]
ylabel=top2[1]

st.pyplot(plot_dr(model_lr, X_train_final, y_train_final, title_lr, xlabel, ylabel))
st.pyplot(plot_dr(model_dt, X_train_final, y_train_final, title_dt, xlabel, ylabel))
st.pyplot(plot_dr(model_knn, X_train_final, y_train_final, title_knn, xlabel, ylabel))


st.write("### ROC-кривые для разных моделей")
st.pyplot(plot_roc_curve())

st.write("### Оценка качества классификации")
st.dataframe(roc_auc_df.round(3))