import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

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

# Инициализация модели логистической регрессии
model = LogisticRegression()

# Обучение модели на тренировочных данных
model.fit(X_train, y_train)

# Предсказания на тренировочной выборке
y_train_pred = model.predict(X_train)

# Предсказания на тестовой выборке
y_test_pred = model.predict(X_test)

# Вычисление метрик для тренировочной выборки
train_report = classification_report(y_train, y_train_pred, output_dict=True)

# Вычисление метрик для тестовой выборки
test_report = classification_report(y_test, y_test_pred, output_dict=True)

# Вывод метрик
print("Метрики на тренировочной выборке:")
get_report(train_report)

print("\nМетрики на тестовой выборке:")
get_report(test_report)

""" Выводы: 
        1. Модель демонстрирует стабильную точность 80% как на тренировочной, так и на тестовой выборках.
         Это указывает на то, что модель хорошо справляется с задачей классификации в целом.
        2. Класс 1 (неопасные)
            * Модель правильно идентифицирует 71% всех примеров класса 1. 
              Это может указывать на то, что есть место для улучшения в обнаружении неопасных объектов.
            * Высокий уровень точности означает, что среди всех объектов, классифицированных как класс 1, 
              86% действительно являются неопасными. Это говорит о том, 
              что модель не слишком часто ошибается в своих предсказаниях для этого класса.
            * F1-Score 78% — это значение является гармоническим средним между Recall и Precision
              и подтверждает сбалансированность между этими метриками.
        3. Класс 2 (опасные)
            * Модель хорошо справляется с выявлением опасных объектов, что является положительным моментом для задач, 
              где важно не пропустить опасные случаи.
            * Precision 75% — хотя точность ниже по сравнению с классом 1, она все же достаточно приемлема. 
              Это означает, что среди всех предсказанных опасных объектов 75% действительно являются таковыми.
            * F1-Score 81% — это значение подтверждает хорошую производительность модели для класса 2
              и указывает на сбалансированность Recall и Precision. 
"""

""" Задание №3. Настройка модели """


def evaluate_model(_model, X_train, y_train, X_test, y_test):
    """Функция для обучения модели и оценки ее производительности"""

    # Обучение модели
    _model.fit(X_train, y_train)

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

# Регуляризация L1
model_l1 = LogisticRegression(penalty='l1', solver='liblinear')
print("\n___Регуляризация L1___\n")
evaluate_model(model_l1, X_train, y_train, X_test, y_test)

# Регуляризация L2
model_l2 = LogisticRegression(penalty='l2', solver='liblinear')
print("\n___Регуляризация L2___\n")
evaluate_model(model_l2, X_train, y_train, X_test, y_test)

# Elastic Net (смешанная регуляризация)
model_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)  # l1_ratio определяет соотношение L1 и L2
print("\n___Регуляризация Elastic Net___\n")
evaluate_model(model_en, X_train, y_train, X_test, y_test)

""" Вывод:
        Все три модели (с L1, L2 и Elastic Net) показывают идентичные метрики как на тренировочной, 
        так и на тестовой выборках. 
        Это говорит о том, что выбранные методы регуляризации не влияют на качество модели в данном случае.
"""

# Построение ROC-кривой

# Получение вероятностей предсказания для тестовой выборки
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Вычисление TPR и FPR
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Вычисление AUC
roc_auc = auc(fpr, tpr)

# Построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Линия случайного угадывания
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ложноположительная частота')
plt.ylabel('Истинноположительная частота')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid()
plt.show()

""" Вывод:
        График ROC-кривой позволяет визуально оценить качество классификатора. 
        Чем ближе кривая к верхнему левому углу графика, тем лучше модель различает классы. 
        Значение AUC (площадь под кривой) варьируется от 0 до 1. AUC = 1 означает идеальную модель. 
        Очень странно получается. но вот так :)
"""

""" Задание №4. Настройка гиперпараметров модели """
# Параметры для логистической регрессии
iterations = [100, 200, 300, 500]
C_values = [0.1, 0.5, 1.0, 3.0, 5.0]

for iters in iterations:
    for C in C_values:
        # Создание и обучение модели
        model = LogisticRegression(max_iter=iters, C=C)
        print(f"\n___Количество итераций {iters} и штраф {C}___\n")
        evaluate_model(model, X_train, y_train, X_test, y_test)


""" Выводы:
    1. Все модели показывают одинаковые метрики на тренировочной и тестовой выборках, 
       что указывает на хорошую обобщающую способность. Это может свидетельствовать о том, 
       что модель не переобучается и хорошо справляется с задачей классификации.
    2. Штрафы от 0.1 до 5.0: Независимо от значения штрафа, метрики остаются неизменными, 
       что может указывать на то, что модель не чувствительна к изменениям штрафа в данном диапазоне. 
       Это может быть связано с тем, что данные достаточно сбалансированы или
       что модель уже оптимизирована для текущих условий.
"""

