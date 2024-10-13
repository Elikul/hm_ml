import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/music_albums.csv')

# Предварительный анализ данных
print(df.head(20))
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Анализ целевой переменной
plt.figure(figsize=(10, 6))
df['popularity'].hist(bins=30)
plt.title('Распределение популярности альбомов')
plt.xlabel('Популярность')
plt.ylabel('Частота')
plt.grid(False)
plt.show()

# Анализ категориальных признаков
print(df['name'].value_counts().head(10))
print(df['release_date'].value_counts().head(10))
print(df['artists'].value_counts().head(10))
print(df['total_tracks'].value_counts().head(10))