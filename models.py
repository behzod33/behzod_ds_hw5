import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

def train_models(X_train, y_train, top2):
    """Обучение моделей только на топ-2 признаках."""
    X_train_top2 = X_train[top2]  # Используем только топ-2 признака

    models = {
        "Logistic Regression": LogisticRegression(max_iter=565),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(max_depth=5)
    }

    for name, model in models.items():
        model.fit(X_train_top2, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    """Оценка моделей и расчет ROC-кривых."""
    results = []
    fprs, tprs, aucs = {}, {}, {}

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_value = auc(fpr, tpr)

        results.append({"Модель": name, "Test AUC": auc_score})
        fprs[name] = fpr
        tprs[name] = tpr
        aucs[name] = auc_value

    return pd.DataFrame(results), fprs, tprs, aucs
