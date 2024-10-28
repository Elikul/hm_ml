import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/nearest_earth_objects.csv')


# Построение ящика с усами
numerical_features = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']

figure, axes = plt.subplots(2, 3, figsize=(16, 8))
figure.suptitle('Ящики с усами для числовых признаков')

axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.boxplot(data=df[feature], ax=axes[i])
    axes[i].set_title(feature)
    axes[i].set_ylabel('')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/boxplot_class_numerical_features.png')
plt.show()

# Ящик с усами позволяет визуализировать распределение и выбросы для различных числовых признаков:
# абсолютной светимости, минимального и максимального диаметра, относительной скорости и расстояния пропуска.
# Для absolute_magnitude и relative_velocity наблюдаются выбросы,
# которые могут указывать на наличие аномально ярких или быстрых астероидов.
# estimated_diameter_min и estimated_diameter_max показывают широкий диапазон значений,
# что также подтверждает наличие крупных объектов среди общего числа.
# miss_distance не имеет выбросов,
# что может быть связано с редкими случаями близкого прохождения астероидов.


# Построение диаграммы распределения для orbiting_body
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='orbiting_body')
plt.title('Распределение по планетам')
plt.xlabel('Планета')
plt.ylabel('Количество астероидов')
plt.savefig('plots/countplot_orbiting_body.png')
plt.show()

# Все рассматриваемые объекты вращаются вокруг Земли, что ожидаемо, так как мы рассматриваем ближайшие земные объекты.

# Построение графика для целевой переменной
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='is_hazardous')
plt.title('Количество опасных и безопасных астероидов')
plt.xlabel('Неопасный (False) / Опасный (True)')
plt.ylabel('Количество')
plt.savefig('plots/countplot_is_hazardous.png')
plt.show()


# График демонстрирует соотношение между опасными (True) и безопасными (False) астероидами.
# Явное преобладание безопасных астероидов указывает на то, что большинство наблюдаемых объектов не представляют угрозу для Земли.
# Однако наличие значительного числа опасных объектов подчеркивает важность мониторинга и изучения N.E.O.
# для предотвращения потенциальных угроз.
# Это соотношение может быть полезным для дальнейшего анализа риска и разработки стратегий по защите от возможных столкновений.

# По ящикам с усами можно заметить, что числовые характеристики как absolute_magnitude,
# estimated_diameter_min, estimated_diameter_max, relative_velocity имеют дисбаланс. Рассмотрим эти признаки детальнее.

def plot_histograms(data, columns, title, xlabel, ylabel, bins, save_path, colors=['orange']):
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


# Построение гистограммы для абсолютной светимости
columns = ['absolute_magnitude']
plot_histograms(df, columns, 'Распределение абсолютной величины астероидов треков',
                'Абсолютная величина', 'Количество', bins=10,
                save_path='plots/absolute_magnitude_distribution.png')

# График показывает распределение абсолютной величины астероидов.
# Наличие пиков может указывать на определенные диапазоны величин, где сосредоточено большее количество объектов.
# Если наблюдаются длинные хвосты, это может свидетельствовать о наличии выбросов.

# Построение гистограммы для минимального и максимального диаметра
columns = ['estimated_diameter_min', 'estimated_diameter_max']
plot_histograms(df, columns, 'Распределение минимального диаметра астероидов',
                'Минимальный диаметр (км)', 'Количество', colors=['orange', 'green'], bins=10,
                save_path='plots/estimated_diameter_distribution.png')

# График распределения минимального и максимального диаметра показывает,
# как часто встречаются астероиды разных размеров.
# Наличие пиков может указывать на типичные размеры астероидов,
# а также на возможные выбросы в виде очень маленьких или очень больших объектов.
# Кривая плотности (kde) подчеркивает, что распределение имеет правостороннюю асимметрию,
# с более высокой концентрацией меньших объектов.

# Построение гистограммы для относительной скорости
columns = ['relative_velocity']
plot_histograms(df, columns, 'Распределение относительной скорости астероидов',
                'Относительная скорость (км/ч)', 'Частота', bins=10,
                save_path='plots/relative_velocity_distribution.png')

# График относительной скорости показывает, как быстро астероиды движутся относительно Земли.
# Наличие пиков может указывать на определенные скорости, которые чаще всего встречаются среди наблюдаемых объектов.
