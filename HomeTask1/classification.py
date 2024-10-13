import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/nearest_earth_objects.csv')

# Предварительный анализ данных
print(df.head(20))
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Анализ целевой переменной
target_counts = df['is_hazardous'].value_counts()

plt.figure(figsize=(10, 6))
target_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Распределение потенциально опасных астероидов')
plt.xlabel('Потенциально опасный астероид')
plt.ylabel('Количество')
plt.grid(False)
plt.show()

# Анализ категориальных признаков
print(df['name'].value_counts().head(10))
print(df['estimated_diameter_min'].value_counts().head(10))
print(df['estimated_diameter_max'].value_counts().head(10))
print(df['relative_velocity'].value_counts().head(10))