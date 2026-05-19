import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# TotalCharges has some spaces instead of NaN — catch them here
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    binary_cols = [c for c in df.columns if df[c].nunique() == 2 and df[c].dtype == object]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    df = pd.get_dummies(df, drop_first=True)
    return df


def split_features(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler
