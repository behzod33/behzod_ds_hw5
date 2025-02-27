import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Загрузка данных из CSV."""
    try:
        column_names = {0: 'ID',
         1: 'RI',
         2: 'Na',
         3: 'Mg',
         4: 'Al',
         5: 'Si',
         6: 'K',
         7: 'Ca',
         8: 'Ba',
         9: 'Fe',
         10: 'glass_type'}
        
        df = pd.read_csv(path, header=None)
        df = df.rename(column_names, axis=1)
        df = df.dropna(subset=["glass_type"])

        df['target'] = df['glass_type'].map({
            2: 0,
            5: 0,
            6: 0,
            1: 1,
            3: 1,
            7: 1
        })
        
        df = df.drop(columns=["ID", "glass_type"])
        df.loc[df["target"] == 0] = df.loc[df["target"] == 0].fillna(df.loc[df["target"] == 0].mean())
        df.loc[df["target"] == 1] = df.loc[df["target"] == 1].fillna(df.loc[df["target"] == 1].mean())
        
        return df.select_dtypes(include=np.number)
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

