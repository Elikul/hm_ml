import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import optuna

""" Задание 1. Предварительная подготовка данных """
# Загрузка данных
data = pd.read_csv('../data/nearest_earth_objects.csv')
print(data.head(20))
print(data.dtypes)

# Проверка дубликатов
duplicates = data.duplicated().sum()
print(f'Количество дубликатов: {duplicates}')
if duplicates > 0:
    data.drop_duplicates(inplace=True)
    print('Дубликаты удалены.')

# Заполнение пропусков
data.fillna({
    'absolute_magnitude': 0,
    'estimated_diameter_min': 0,
    'estimated_diameter_max': 0,
    'relative_velocity': 0,
    'miss_distance': 0,
    'orbiting_body': 'other'
}, inplace=True)

# График количества значений по классу is_hazardous
def is_hazardous_plot(_data):
    fig = px.pie(_data, names='is_hazardous', title='Count Of is_hazardous')
    fig.show()

is_hazardous_plot(data)

# Видно, что у нас дисбаланс классов, который нужно исправить

# посчитать количество уникальных значений
unique_counts = data.select_dtypes("object").nunique()
print(unique_counts)

# Определение числовых и категориальных признаков. Заполнить пропуски в данных
# удалить идентификатор, так как он не несёт никакой полезной для нас информации
data.drop(columns=['neo_id'], inplace=True)
numerical_cols = data.select_dtypes(include=['float64', 'int']).columns
data[numerical_cols] = data[numerical_cols].fillna(0)
print(numerical_cols)

categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna('other')
print(categorical_cols)

# Создание экземпляра StandardScaler
scaler = StandardScaler()

# Стандартизация числовых признаков
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
print(data[numerical_cols])

# Создание экземпляра LabelEncoder
label_encoder = LabelEncoder()

# Применение LabelEncoder к каждому категориальному столбцу
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

print(data.head(20))

# Разделить на тренировочную и обучающую выборки
target_column = "is_hazardous"
features = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'miss_distance', 'relative_velocity']
X = data[features]
y = data[target_column]

print(X.shape, y.shape)

# Избавиться от дисбаланса классов
# SMOTE создает синтетические примеры, основываясь на существующих данных
smt = SMOTE()
X_resampled, y_resampled = smt.fit_resample(X, y)
print(X_resampled.shape, y_resampled.shape)

is_hazardous_plot(y_resampled)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.25, stratify=y_resampled, random_state=42
)
print(X_train.head(2))

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(X_train.head())
print(X_test.head())

''' Задание №2. Обучение модели и вычисление метрик '''

def get_report(report):
    """Функция для вывода метрик"""
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Класс 1 (неопасные): Recall: {report['False']['recall']:.2f}, Precision: {report['False']['precision']:.2f}, F1-Score: {report['False']['f1-score']:.2f}")
    print(f"Класс 2 (опасные): Recall: {report['True']['recall']:.2f}, Precision: {report['True']['precision']:.2f}, F1-Score: {report['True']['f1-score']:.2f}")

def evaluate(_model, X_train, y_train, X_test, y_test):
    """Функция для оценки модели и её производительности"""

    # Предсказания на тренировочной выборке
    y_train_pred = _model.predict(X_train)

    # Предсказания на тестовой выборке
    y_test_pred = _model.predict(X_test)

    # Вычисление метрик для тренировочной выборки
    train_report = classification_report(y_train, y_train_pred, output_dict=True)

    # Вычисление метрик для тестовой выборки
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    # Вывод метрик
    print("Метрики на тренировочной выборке:")
    get_report(train_report)

    print("\nМетрики на тестовой выборке:")
    get_report(test_report)

    return f1_score(y_test, y_test_pred, average='weighted')

# Создание и обучение модели с k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Предсказание на тренировочной и тестовой выборке, вычисление метрик
evaluate(knn, X_train, y_train, X_test, y_test)

'''Выводы:
        1. Модель демонстрирует высокую точность как на тренировочной, так и на тестовой выборках, 
           что указывает на хорошую обобщающую способность.
        2. Класс 1 (неопасные)
            * Recall показывает, что модель хорошо находит большинство объектов класса "неопасные", 
              хотя на тестовой выборке наблюдается небольшое снижение.
            * Precision остается высоким, что означает, 
              что большинство предсказанных объектов действительно относятся к классу "неопасные".
            * F1-Score также высок, что указывает на сбалансированную производительность между Recall и Precision.
        3. Класс 2 (опасные)
            * Модель показывает отличные результаты по Recall для класса "опасные", что говорит о том, 
              что она успешно идентифицирует почти все объекты этого класса.
            * Снижение Precision на тестовой выборке может указывать на то, 
              что модель иногда ошибочно классифицирует объекты из класса "неопасные" как "опасные".
            * F1-Score остается высоким, что подтверждает хорошую сбалансированность модели.
'''

# Список значений k для тестирования
k_values = range(1, 21)
accuracies = []

# Обучение модели и оценка точности для каждого значения k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acr = accuracy_score(y_test, y_pred)
    accuracies.append(acr)
    print(f'Количество соседей (k): {k}, Точность модели на тестовых данных: {acr:.2f}\n')

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Зависимость точности от количества соседей (k)')
plt.xlabel('Количество соседей (k)')
plt.ylabel('Точность')
plt.xticks(k_values)
plt.grid()
plt.show()

'''Выводы:
        * При использовании 1 соседа (k=1) модель показывает наивысшую точность — 0.93. 
          Это свидетельствует о том, 
          что использование одного ближайшего соседа позволяет модели хорошо классифицировать данные.
        * С увеличением количества соседей (k) наблюдается постепенное снижение точности. 
          Например, при k=2 точность составляет 0.92, а при k=3 — 0.91. Это снижение продолжает наблюдаться до k=15, 
          где точность стабилизируется на уровне 0.88.
        * Начиная с k=6, точность остается относительно стабильной, колеблясь между 0.87 и 0.90.
          Это указывает на то, что увеличение числа соседей не всегда приводит к улучшению качества предсказаний.
        * На основании полученных данных оптимальным значением для k является 1 или 2,
          так как они обеспечивают наивысшую точность. 
          Однако использование одного соседа может привести к переобучению модели, особенно если данные имеют шум.
        * При увеличении k модель становится менее чувствительной к шуму в данных,
          что может быть полезно для повышения обобщающей способности, но может также привести к потере информации.
'''

''' Задание №3. Настройка гиперпараметров модели '''


def objective(trial):
    # Подбор гиперпараметров
    n_neighbors = trial.suggest_int('n_neighbors', 1, 15)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
    algorithm = trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'auto'])

    # Обучение модели
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm, n_jobs=-1)
    model.fit(X_train, y_train)

    # Вычисление метрик
    return evaluate(model, X_train, y_train, X_test, y_test)

# Создание и оптимизация исследования Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5, n_jobs=-1)

# Получение лучших параметров
best_params = study.best_params
print("Лучшие параметры:", best_params)

'''Выводы:
        1. Trials:
            * Trial 1: Лучшая точность на тестовой выборке составила 0.91 с параметрами n_neighbors=2, metric='chebyshev', algorithm='kd_tree'.
            * Trial 2: Значение метрики составило только 0.8734, что значительно ниже.
            * Trial 3: Точность на тестовой выборке составила 0.90 с параметрами n_neighbors=7, metric='manhattan', algorithm='auto'.
            * Trial 4: Значение метрики составило 0.8849, также ниже по сравнению с лучшим результатом.
        2. Модель с параметрами n_neighbors=4, metric='manhattan' и algorithm='ball_tree' 
           показала наилучшие результаты как на тренировочной, так и на тестовой выборках.
        3. Высокие значения recall и precision для обоих классов указывают на хорошую способность модели 
           различать опасные и неопасные объекты.
'''

# Обучение модели с лучшими метриками
best_model = KNeighborsClassifier(**best_params)
best_model.fit(X_train, y_train)

# Предсказание на тренировочной и тестовой выборке, вычисление метрик
evaluate(best_model, X_train, y_train, X_test, y_test)

'''Выводы:
        1. Высокая точность (accuracy) на тренировочной выборке (95%) и хорошая на тестовой выборке (92%) указывают на то,
           что модель хорошо обобщает данные и не переобучена.
        2. Класс 1 (неопасные):
            * На тренировочной выборке модель показывает высокий recall (93%) и precision (98%),
              что говорит о том, что она успешно идентифицирует большинство неопасных объектов
              и минимизирует ложные срабатывания.
            * На тестовой выборке recall снижается до 88%, 
              что может указывать на некоторую потерю информации или сложность в распознавании неопасных объектов 
              в новых данных, однако precision остается высоким (95%).
        3. Класс 2 (опасные):
            * Модель показывает отличные результаты как по recall (98% на тренировочной и 96% на тестовой выборках), 
              так и по precision (93% на тренировочной и 89% на тестовой). Это свидетельствует о том, 
              что модель надежно распознает опасные объекты.
'''