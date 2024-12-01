import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

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

""" Задание №2. Обучение модели логистической регрессии и вычисление метрик """

def get_report(report):
    """Функция для вывода метрик"""
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Класс 1 (неопасные): Recall: {report['False']['recall']:.2f}, Precision: {report['False']['precision']:.2f}, F1-Score: {report['False']['f1-score']:.2f}")
    print(f"Класс 2 (опасные): Recall: {report['True']['recall']:.2f}, Precision: {report['True']['precision']:.2f}, F1-Score: {report['True']['f1-score']:.2f}")

def evaluate_model(_model, X_train, y_train, X_test, y_test):
    """Функция для оценки модели производительности"""
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


# Создание модели
model = DecisionTreeClassifier()

# Обучение модели и вычисление метрик
model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train, X_test, y_test)

""" Выводы:
        1. Модель идеально классифицировала все примеры в тренировочной выборке, что указывает на отсутствие ошибок.
        2. Модель показала высокую точность на тестовой выборке, 
           однако наблюдается небольшое снижение по сравнению с тренировочной выборкой.
        3. Идеальные результаты на тренировочной выборке могут указывать на переобучение модели, 
           поскольку она не справляется с некоторыми аспектами данных, 
           которые не были представлены в тренировочном наборе.
        4. Несмотря на небольшое снижение производительности на тестовой выборке, 
           модель все еще показывает высокие показатели качества, 
           что свидетельствует о ее способности обобщать информацию.
"""

""" Задание №3. Настройка гиперпараметров модели """

# Определение параметров для GridSearch
param_grid = {
    'max_depth': range(1, 21),  # Максимальная глубина дерева
    'min_samples_split': range(2, 21)  # Минимальное количество образцов для разделения узла
}

# Создание модели DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# Подбор гиперпараметров с помощью GridSearchCV
grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Получение лучших параметров
best_params = grid_search.best_params_
print(f"Лучшие параметры: {best_params}")

# Обучение модели с лучшими параметрами
best_model = grid_search.best_estimator_

# Вычисление и вывод метрик
evaluate_model(best_model, X_train, y_train, X_test, y_test)

""" Выводы:
        1. Модель деревьев решений с параметрами глубины 20 и минимальным количеством образцов 2 демонстрирует 
           хорошие результаты как на тренировочной, так и на тестовой выборках. 
           Общая точность составляет 89% на тренировочной выборке и 87% на тестовой. 
           Это свидетельствует о том, что модель хорошо обобщает данные и не переобучается.
        2. Для класса "неопасные" модель показывает высокую точность (precision), 
           что означает, что большинство предсказанных "неопасных" объектов действительно таковыми являются. 
           Однако recall несколько ниже, что указывает на то, 
           что некоторые "неопасные" объекты могут быть неправильно классифицированы как "опасные".
           Снижение метрик на тестовой выборке относительно тренировочной также говорит о 
           некоторой потере качества обобщения.
        3. Для класса "опасные" модель демонстрирует очень высокий recall, что говорит о том,
           что большинство "опасных" объектов правильно классифицируются. 
           Однако precision немного ниже, что может указывать на наличие ложноположительных результатов. 
           Снижение метрик на тестовой выборке также наблюдается, но в целом модель сохраняет хорошее качество.
"""