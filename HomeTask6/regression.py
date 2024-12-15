import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna


'''Задание 1. Предварительная подготовка данных'''

df = pd.read_csv('../data/music_albums.csv')

print(df.dtypes)
print(df.head(20))

# Проверка дубликатов
duplicates = df.duplicated().sum()
print(f'Количество дубликатов: {duplicates}')
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print('Дубликаты удалены.')

# посчитать количество уникальных значений
unique_counts = df.select_dtypes("object").nunique()
print(unique_counts)

# Определение числовых и категориальных признаков. Заполнить пропуски в данных
# удалить идентификатор, так как он не несёт никакой полезной для нас информации
df.drop(columns=['id'], inplace=True)
numerical_cols = df.select_dtypes(include=['float64', 'int']).columns
df[numerical_cols] = df[numerical_cols].fillna(0)
print(numerical_cols)

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna('other')
print(categorical_cols)

# Создание экземпляра StandardScaler
scaler = StandardScaler()

# Стандартизация числовых признаков
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df[numerical_cols])

# Создание экземпляра LabelEncoder
label_encoder = LabelEncoder()

# Применение LabelEncoder к каждому категориальному столбцу
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Разделить на тренировочную и обучающую выборки
# Из прошлого домашнего задания, где я делала визуализацию, я выявила признаки,
# которые являются более важными, поэтому их мы и берём в модель
target_column = "popularity"
features = ['t_dur0', 't_dur1', 't_dur2', 't_speech0', 't_speech1', 't_speech2', 't_acous0', 't_acous1',
            't_acous2', 't_ins0', 't_ins1', 't_ins2', 't_live0', 't_live1', 't_live2', 't_val0', 't_val1',
            't_val2', 't_tempo0', 't_tempo1', 't_tempo2', 't_sig0', 't_sig1', 't_sig2', 't_dance0',
            't_dance1', 't_dance2', 't_energy0', 't_energy1', 't_energy2', 't_key0', 't_key1', 't_key2',
            't_mode0', 't_mode1', 't_mode2']
X = df[features]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
X_train.head(2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

'''Задание №2. Обучение модели линейной регрессии и вычисление метрик'''

# Создание модели SVR с параметрами по умолчанию
model = SVR(verbose=True)

# Обучение модели
model.fit(X_train, y_train)


# Предсказание на тренировочной и тестовой выборке
def evaluate(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Вычисление метрик качества
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Вывод результатов
    print(f"Тренировочная MSE: {mse_train}, MAE: {mae_train}, R^2: {r2_train},")
    print(f"Тестовая MSE: {mse_test}, MAE: {mae_test}, R^2: {r2_test}")

    return mse_test

evaluate(model, X_train, y_train, X_test, y_test)

'''Выводы:
       1. Тренировочная MSE ниже тестовой, что может указывать на некоторую степень переобучения модели. 
          Модель хорошо справляется с обучающим набором данных, 
          но ее способность обобщать на новых данных (тестовом наборе) несколько снижается.
       2. MAE также показывает, что модель делает более точные прогнозы на обучающем наборе, чем на тестовом. 
          Разница в значениях MAE подтверждает, что модель может быть менее надежной при предсказании новых данных.
       3. Значения R² близки к 0.5, что указывает на то, 
          что модель объясняет примерно половину вариации целевой переменной как на тренировочном, 
          так и на тестовом наборах данных. Это низкий уровень объясненной дисперсии, что говорит о том, 
          что модель не полностью понимает зависимость между признаками и целевой переменной.
'''

''' Задание №3. Настройка гиперпараметров модели '''


# Определение целевой функции для Optuna
def objective(trial):
    # Выбор гиперпараметров
    kernel = trial.suggest_categorical('kernel', ['poly', 'rbf'])
    C = trial.suggest_float('C', 1, 10)
    gamma = trial.suggest_float('gamma', 0.01, 1)

    # Создание и обучение модели
    model = SVR(kernel=kernel, C=C, gamma=gamma, verbose=True)
    model.fit(X_train, y_train)

    # Вычисление метрик
    return evaluate(model, X_train, y_train, X_test, y_test)


# Создание объекта исследования Optuna и оптимизация гиперпараметров
study = optuna.create_study()
study.optimize(objective, n_trials=1, n_jobs=-1)

# Получение лучших параметров
best_params = study.best_params
print("Лучшие параметры:", best_params)

# Обучение модели с лучшими гиперпараметрами
best_model = SVR(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], verbose=True)
best_model.fit(X_train, y_train)

evaluate(best_model, X_train, y_train, X_test, y_test)

'''Выводы:
        1. Использовала одну попытку поиска и маленькие значения С и GAMMA, 
           потому что не хватало мощностей компьютера. SVR обучалась очень очень долго, 
           а потом ноутбук просто вылетал. Пробовала считать на GPU в Google Colab, 
           но там бесплатные мощности заканчивались раньше, чем успевала обучиться модель. 
           Поэтому решила попробовать хотя бы этот вариант.
        2. Точность модели на тренировочных данных
                * MSE 0.1464 - относительно низкое значение, 
                  указывающее на хорошую точность модели при предсказании на тренировочных данных.
                * MAE 0.2119 - также относительно низкое значение, подтверждающее хорошую точность.
                * R^2 0.8536 - высокий коэффициент указывает на то, что модель хорошо объясняет вариацию в данных.
        3. Точность модели на тестовых данных
                * MSE 0.3114 - значение заметно выше, чем на тренировочных данных, 
                  что может указывать на некоторую степень обучения.
                * MAE 0.3549 - значение выше, чем на тренировочных данных, что подтверждает наличие обучения.
                * R^2 0.6886 - относительно высокий коэффициент, он заметно ниже, чем на тренировочных данных, 
                  что также указывает на возможность объяснять вариацию в данных.
        4. Модель SVR показывает хорошую точность на тренировочных данных, 
           но наблюдается некоторое снижение метрик на тестовой, что говорит о том, 
           что всё-таки следует ещё подбирать гиперпараметры. Но к сожалению,
           не хватает мощностей на обучение с гипермпараметрами побольше.
'''