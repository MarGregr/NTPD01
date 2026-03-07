import pandas as pd
from sklearn.datasets import load_iris


#Zadanie 1
print("Zadanie 1")

#Załadowanie zbioru iris z biblioteki scikit-learn
data = load_iris()

#Konwertujemy na format DataFrame (Pandas), dla wygodniejszej analizy danych
df = pd.DataFrame(data.data, columns=data.feature_names)

#Krótka analiza danych
print("Pierwsze 5 wierszy danych:")
print(df.head())

print("\nInformacje o rozmiarze macierzy (wiersze, kolumny):", df.shape)

print("\nTypy danych i brakujące wartości:")
print(df.info())

print("\nPodstawowe statystyki opisowe:")
print(df.describe())


#Zadanie 2
print("\nZadanie 2")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df
y = data.target

#Podział na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

#Wytrenowanie modelu
model.fit(X_train, y_train)

#Predykcja
y_pred = model.predict(X_test)

#Metryki
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nSzczegółowy raport:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

#Zadnie 3
print("\nZadanie 3")

import joblib

model_filename = 'model_v1.joblib'

#Zapisywanie modelu do pliku
joblib.dump(model, model_filename)

print(f"Model został zapisany do pliku: {model_filename}")