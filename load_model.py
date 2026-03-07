import joblib
import pandas as pd

# 1. Wczytanie modelu
loaded_model = joblib.load('model_v1.joblib')
print("Model został wczytany poprawnie")

#Definicja nazw cech identycznych jak w zbiorze Iris
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#Przykładowy nowy rekord
sample_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)

#Wykonanie predykcji
prediction = loaded_model.predict(sample_data)
#Mapowanie wyniku na nazwę gatunku irysa
species = ['setosa', 'versicolor', 'virginica']

print(f"\nWynik predykcji dla danych:\n{sample_data}")
print(f"Gatunek: {species[prediction[0]]}")