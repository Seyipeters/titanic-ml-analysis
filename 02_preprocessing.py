"""Stage 2: Preprocessing"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("../data/titanic.csv")
target = "survived"
if target not in df.columns: target = df.columns[-1]
X = df.drop(columns=[target])
Y = df[target]
X = X.loc[:, X.isnull().mean() < 0.5]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.select_dtypes(include=["object","category"]).columns
            if X[c].nunique() <= 50]
for col in num_cols: X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str).fillna("missing"))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols + cat_cols].values.astype(float))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
