import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train, top2, hyperparams=None):
    """Обучение только выбранных моделей с гиперпараметрами."""
    X_train_top2 = X_train[top2]

    # Проверяем, переданы ли гиперпараметры
    if not hyperparams:
        return {}

    models = {}

    # Добавляем модели, только если они были выбраны пользователем
    if "Logistic Regression" in hyperparams:
        models["Logistic Regression"] = LogisticRegression(**hyperparams["Logistic Regression"])
    if "KNN" in hyperparams:
        models["KNN"] = KNeighborsClassifier(**hyperparams["KNN"])
    if "Decision Tree" in hyperparams:
        models["Decision Tree"] = DecisionTreeClassifier(**hyperparams["Decision Tree"])

    # Обучаем выбранные модели
    for name, model in models.items():
        model.fit(X_train_top2, y_train)

    return models

def tune_hyperparameters_randomized(models, X_train, y_train, param_grid):
    """Подбор гиперпараметров для моделей с использованием RandomizedSearchCV."""
    best_models = {}

    for model_name, model in models.items():
        if model_name in param_grid:
            random_search = RandomizedSearchCV(model, param_grid[model_name], n_iter=10, cv=5, n_jobs=-1, scoring='accuracy')
            random_search.fit(X_train, y_train)
            best_models[model_name] = random_search.best_estimator_ 

    return best_models

def tune_hyperparameters(models, X_train, y_train, param_grid):
    """Подбор гиперпараметров для моделей с использованием GridSearchCV."""
    best_models = {}

    for model_name, model in models.items():
        if model_name in param_grid:
            grid_search = GridSearchCV(model, param_grid[model_name], cv=5, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_ 

    return best_models



def plot_confusion_matrix(y_true, y_pred, model_name):
    """Построение матрицы ошибок."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    return fig

def get_classification_metrics(y_true, y_pred):
    """Формирование отчёта о классификации в виде DataFrame для Precision, Recall, F1-score."""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Извлекаем только нужные метрики (Precision, Recall, F1-score)
    metrics = {
        "Precision": {},
        "Recall": {},
        "F1-score": {}
    }
    
    # Перебираем классы 0 и 1, а также макро и взвешенные метрики
    for key in ['0', '1']:
        metrics["Precision"][key] = report[key]["precision"]
        metrics["Recall"][key] = report[key]["recall"]
        metrics["F1-score"][key] = report[key]["f1-score"]

    # Преобразуем в DataFrame для отображения
    df_report = pd.DataFrame(metrics)
    return df_report.round(2)


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Оценка моделей: расчет Train AUC, Test AUC и разницы AUC."""
    results = []
    fprs, tprs, aucs = {}, {}, {}

    for name, model in models.items():
        # AUC на тренировочных данных
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_proba)

        # AUC на тестовых данных
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)

        # Разница между Train AUC и Test AUC
        auc_diff = abs(train_auc - test_auc)

        # ROC-кривые для тестовых данных
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auc_value = auc(fpr, tpr)

        results.append({
            "Модель": name,
            "Train AUC": round(train_auc, 3),
            "Test AUC": round(test_auc, 3),
            "Разница AUC": round(auc_diff, 3)
        })

        fprs[name] = fpr
        tprs[name] = tpr
        aucs[name] = auc_value

    return pd.DataFrame(results), fprs, tprs, aucs

def plot_f1_scores(models, X_test, y_test, top2):
    """Построение графика F1-score для всех моделей."""
    f1_scores = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test[top2])
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_scores[model_name] = report["weighted avg"]["f1-score"]

    # Построение графика
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(f1_scores.keys()), list(f1_scores.values()), color="blue", alpha=0.7)
    ax.set_xlabel("F1-score")
    ax.set_title("F1-score для моделей")
    ax.set_xlim(0, 1)
    
    for i, v in enumerate(f1_scores.values()):
        ax.text(v + 0.02, i, str(round(v, 3)), va="center", fontsize=10)

    return fig

