import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, y_train, top2, hyperparams=None):
    """Обучение моделей только на топ-2 признаках с возможностью настройки гиперпараметров."""
    X_train_top2 = X_train[top2]

    # Значения по умолчанию для гиперпараметров
    default_params = {
        "Logistic Regression": {"max_iter": 565, "C": 1.0},
        "KNN": {"n_neighbors": 3, "weights": "uniform"},
        "Decision Tree": {"max_depth": 5, "criterion": "gini"}
    }

    # Объединение переданных гиперпараметров с дефолтными
    if hyperparams:
        for model_name, params in hyperparams.items():
            default_params[model_name].update(params)

    # Создаем модели с заданными параметрами
    models = {
        "Logistic Regression": LogisticRegression(**default_params["Logistic Regression"]),
        "KNN": KNeighborsClassifier(**default_params["KNN"]),
        "Decision Tree": DecisionTreeClassifier(**default_params["Decision Tree"])
    }

    # Обучаем модели
    for name, model in models.items():
        model.fit(X_train_top2, y_train)

    return models


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
