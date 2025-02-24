import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Загрузка данных из CSV."""
    try:
        return pd.read_csv(path)
    except:
        return None
    
def get_top_features(df, target="target", top_n=3):
    """Определяет топ-N признаков, наиболее коррелирующих с целевой переменной."""
    cor_mat = df.corr().round(2)
    top_features = cor_mat[target].abs().sort_values(ascending=False).index[1:top_n + 1].tolist()
    return top_features


def preprocess_data(df):
    """Предобработка данных: разделение на train/test и стандартизация."""
    X = df.drop(columns="target")
    y = df["target"]

    # Определение топ-2 и топ-3 признаков
    cor_mat = X.corr().round(2)
    top3 = cor_mat.abs().sum().sort_values(ascending=False)[:3].index.tolist()
    top2 = top3[:2]

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    # Стандартизация
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test, top2, top3

