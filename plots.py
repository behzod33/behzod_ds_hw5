import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mlxtend.plotting import plot_decision_regions

def plot_correlation_matrix(df):
    """Построение матрицы корреляции."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Матрица корреляции")
    return fig

def plot_3d_scatter(df, top3):
    """Построение 3D Scatter Plot."""
    color_map = {"0": "orange", "1": "blue"}
    fig = px.scatter_3d(
        df,
        x=top3[2],
        y=top3[1],
        z=top3[0],
        color=df["target"].astype(str),
        color_discrete_map=color_map,
        title="3D Scatter Plot для бинарной классификации"
    )
    return fig

def plot_decision_boundaries(model, X, y, title, xlabel, ylabel):
    """Построение границ решений."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X, y, clf=model, legend=2, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig

def plot_roc_curve(fprs, tprs, aucs):
    """Построение ROC-кривых."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    for name, fpr in fprs.items():
        ax.plot(fpr, tprs[name], label=f"{name} (AUC = {aucs[name]:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="dotted")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC-кривые для разных моделей")
    ax.legend(loc="lower right")
    ax.grid(True)
    return fig
