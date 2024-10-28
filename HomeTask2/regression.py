import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/music_albums.csv')

# Построение гистограммы
plt.figure(figsize=(10, 6))
plt.hist(df['popularity'], bins=20, color='blue', alpha=0.7)
plt.title('Распределение популярности альбомов')
plt.xlabel('Популярность')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.75)
plt.savefig('plots/popularity_distribution.png')
plt.show()

# Гистограмма показывает, что большинство альбомов имеют среднюю популярность, с небольшим количеством альбомов
# которые являются либо очень популярными, либо непопулярными.
# Это может свидетельствовать о том, что большинство музыкальных произведений находится в среднем диапазоне популярности.

# Построение ящиков с усами для числовых признаков
numerical_features = ['t_dur0', 't_dur1', 't_dur2', 't_speech0', 't_speech1', 't_speech2', 't_acous0', 't_acous1',
                      't_acous2', 't_ins0', 't_ins1', 't_ins2', 't_live0', 't_live1', 't_live2', 't_val0', 't_val1',
                      't_val2', 't_tempo0', 't_tempo1', 't_tempo2', 't_sig0', 't_sig1', 't_sig2', 't_dance0',
                      't_dance1', 't_dance2', 't_energy0', 't_energy1', 't_energy2', 't_key0', 't_key1', 't_key2',
                      't_mode0', 't_mode1', 't_mode2']

figure, axes = plt.subplots(3, 4, figsize=(16, 8))
figure.suptitle('Ящики с усами для числовых признаков')

axes = axes.flatten()

for i in range(0, len(numerical_features), 3):
    features_to_plot = numerical_features[i:i + 3]

    sns.boxplot(data=df[features_to_plot], ax=axes[i // 3])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/boxplots_numerical_features.png')
plt.show()

# Ящики с усами показывают наличие выбросов в характеристиках
# длительности (t_dur), танцевальности (t_dance), энергии (t_energy), тональности (t_key), модальности (t_mode),
# разборчивости слов (t_speech), достоверности (t_acous), инструментальности (t_ins), живучести (t_live),
# позитивности (t_val), времени (t_sig) и темпа (t_tempo).
# Например, некоторые треки имеют очень высокие значения танцевальности,
# что может указывать на их предназначение для клубной музыки или танцевальных мероприятий.
# Наличие выбросов также может свидетельствовать о разнообразии музыкальных стилей в датасете.

# Построение диаграммы для текстового признака artists
df['artists'] = df['artists'].str.rstrip(', ')
filtered_df = df[df['artists'] != 'Various Artists']
top_artists = filtered_df['artists'].value_counts().head(10)
top_artists_df = top_artists.reset_index()
top_artists_df.columns = ['artists', 'album_count']

plt.figure(figsize=(15, 8))
sns.barplot(y='artists', x='album_count', data=top_artists_df, color='green')
plt.title('Топ-10 исполнителей по количеству альбомов')
plt.xticks(rotation=45)
plt.ylabel('Исполнители')
plt.xlabel('Количество альбомов')
plt.savefig('plots/top_artists_distribution.png')
plt.show()


# Столбиковая диаграмма показывает, что несколько исполнителей доминируют в датасете по количеству альбомов.
# Это может указывать на их активность в музыкальной индустрии и популярность среди слушателей.
# Если наблюдается значительный дисбаланс
# (например, один или два исполнителя имеют значительно больше альбомов),
# это может свидетельствовать о тенденциях в музыкальной индустрии.


# По ящикам с усами можно заметить, что числовые характеристики как
# длительности (t_dur), танцевальности (t_dance), разборчивости слов (t_speech), инструментальности (t_ins),
# живучести (t_live), времени (t_sig) и темпа (t_tempo) имеют дисбаланс. Рассмотрим эти признаки детальнее.

def plot_histograms(data, columns, title, xlabel, ylabel, colors, bins, save_path):
    plt.figure(figsize=(12, 8))

    for col, color in zip(columns, colors):
        sns.histplot(data[col], bins=bins, kde=True, color=color, label=col, alpha=0.5)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    plt.show()

plt.figure(figsize=(12, 8))

# Построение гистограмм для t_dance0, t_dance1, t_dance2
dance_columns = ['t_dance0', 't_dance1', 't_dance2']
dance_colors = ['orange', 'blue', 'green']
plot_histograms(df, dance_columns, 'Распределение танцевальности треков', 'Танцевальность',
                'Частота', dance_colors, bins=30, save_path='plots/danceability_multiple_distribution.png')

# Графики показывают распределение танцевальности для трех различных значений.
# Они имеют схожие распределения, это может указывать на то,
# что все версии треков имеют схожие характеристики по танцевальности.
# Если же одно из значений значительно отличается от других,
# это может указывать на уникальные особенности одной из версий.

# Построение гистограмм для t_dur0, t_dur1, t_dur2
dur_columns = ['t_dur0', 't_dur1', 't_dur2']
dur_colors = ['orange', 'blue', 'green']
plot_histograms(df, dur_columns, 'Распределение длительности треков', 'Длительность', 'Частота',
                dur_colors, bins=5, save_path='plots/duration_distribution.png')

# Гистограмма показывает, что длительность треков варьируется от коротких (менее 2 минут) до длинных (более 6 минут).
# Наиболее частые значения длительности находятся в диапазоне от 3 до 4 минут,
# что соответствует стандартной длине большинства поп- и рок-композиций.
# Наличие выбросов в обеих крайностях может указывать на наличие экспериментальных треков или композиций в жанрах,
# таких как классическая музыка или электронная музыка, где длительность может значительно варьироваться.

# Построение гистограмм для t_speech0, t_speech1, t_speech2
speech_columns = ['t_speech0', 't_speech1', 't_speech2']
speech_colors = ['orange', 'blue', 'green']
plot_histograms(df, speech_columns, 'Распределение разборчивости слов в треках', 'Разборчивость слов',
                'Частота', speech_colors, bins=5, save_path='plots/speech_distribution.png')

# Гистограмма разборчивости слов показывает, что большинство треков имеют низкие значения разборчивости (меньше 0.5).
# Это может указывать на преобладание инструментальной музыки или стилей,
# таких как EDM и хип-хоп, где вокал не является основным элементом композиции.
# Высокие значения разборчивости могут быть связаны с песнями, акцентирующими внимание на текстах и вокале.

# Построение гистограмм для t_ins0, t_ins1, t_ins2
ins_columns = ['t_ins0', 't_ins1', 't_ins2']
ins_colors = ['orange', 'blue', 'green']
plot_histograms(df, ins_columns,'Распределение инструментальности треков', 'Инструментальность',
                'Частота', ins_colors, bins=5, save_path='plots/instrumental_distribution.png')

# График инструментальности показывает некоторое количество треков с высокими значениями (ближе к 1.0),
# что указывает на наличие инструментальных композиций в датасете.
# Это может быть характерно для жанров, таких как джаз или классическая музыка, где вокал часто отсутствует.
# Большое количество имеет низкие значения, что может указывать на песни с ярко выраженным вокалом.

# Построение гистограмм для t_live0, t_live1, t_live2
live_columns = ['t_live0', 't_live1', 't_live2']
live_colors = ['orange', 'blue', 'green']
plot_histograms(df, live_columns, 'Распределение живучести треков', 'Живучесть', 'Частота',
                live_colors, bins=5, save_path='plots/liveness_distribution.png')

# Гистограмма живучести показывает, что большинство треков имеют низкие значения (менее 0.3),
# что свидетельствует о преобладании студийных записей над записями живых выступлений.
# Высокие значения живучести (ближе к 1.0) могут указывать на записи живых концертов или выступлений,
# что может быть интересно для слушателей, предпочитающих живую музыку.

# Построение гистограмм для t_sig0, t_sig1, t_sig2
sig_columns = ['t_sig0', 't_sig1', 't_sig2']
sig_colors = ['orange', 'blue', 'green']
plot_histograms(df, sig_columns, 'Распределение временной характеристики треков',
                'Временная характеристика (meter)', 'Частота', sig_colors, bins=5,
                save_path='plots/time_signature_distribution.png')

# График временной характеристики показывает разнообразие размеров в музыкальных произведениях.
# Преобладание стандартных размеров (например, 4/4) говорит о том,
# что большинство треков соответствует популярным музыкальным формам и стилям.
# Однако наличие других размеров может указывать на экспериментальные композиции или жанры,
# такие как прогрессивный рок или джаз.

# Построение гистограмм для t_tempo0, t_tempo1, t_tempo2
tempo_columns = ['t_tempo0', 't_tempo1', 't_tempo2']
tempo_colors = ['orange', 'blue', 'green']
plot_histograms(df, tempo_columns, 'Распределение темпа треков', 'Темп (BPM)', 'Частота',
                tempo_colors, bins=5, save_path='plots/tempo_distribution.png')

# Гистограмма темпа показывает значительное количество треков в диапазоне от 100 до 150 BPM,
# что соответствует большинству популярных музыкальных стилей, таких как поп и танцевальная музыка.
# Выбросы на обоих концах диапазона могут указывать на наличие как очень медленных (менее 60 BPM),
# так и очень быстрых (более 180 BPM) треков, что может быть характерно для определенных жанров,
# таких как техно.
