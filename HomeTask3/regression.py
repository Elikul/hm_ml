import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, Lasso, Ridge, ElasticNet
from sklearn.metrics import root_mean_squared_error, r2_score

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

# Создание и обучение модели
model = SGDRegressor()
model.fit(X_train, y_train)

# Предсказание на тренировочной и тестовой выборке
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Вычисление метрик
r2_train = r2_score(y_train, y_train_pred)
rmse_train = root_mean_squared_error(y_train, y_train_pred)

r2_test = r2_score(y_test, y_test_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

print(f"Тренировочная R^2: {r2_train}, RMSE: {rmse_train}")
print(f"Тестовая R^2: {r2_test}, RMSE: {rmse_test}")

''' Выводы:
* Значения R² как для тренировочного, так и для тестового наборов данных близки к нулю, что указывает на то, 
  что модель объясняет лишь небольшую долю вариации целевой переменной. 
  Это говорит о том, что модель не является хорошей и не может адекватно предсказать результаты.
* Значения RMSE в пределах 0.90 показывают,
  что средняя ошибка предсказаний модели составляет примерно 0.90 единицы от реальных значений. 
  Это также указывает на то, что модель имеет значительные ошибки в предсказаниях.
* Результаты обучения модели SGDRegressor показывают, что она не справляется с задачей предсказания целевой переменной. 
  Низкие значения R² и относительно высокие значения RMSE указывают на необходимость улучшения модели. 
  Так получилось из-за того, что по заданию нужно было использовать SGDRegressor с параметрами по умолчанию, 
  которые не подходят под наши данные. '''

'''Задание №3. Настройка регуляризации модели'''

# Обучение модели линейной регрессии, используя методы регуляризации L1, L2 и ElasticNet
alphas = [0.01, 0.3, 0.5, 0.8]
models = {'Lasso': Lasso(), 'Ridge': Ridge(), 'ElasticNet': ElasticNet()}

results = {}

for model_name, model in models.items():
    for alpha in alphas:
        model.set_params(alpha=alpha)
        model.fit(X_train, y_train)

        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Вычиление метрик
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)

        results[(model_name, alpha)] = {
            'R2 Train': r2_train,
            'R2 Test': r2_test,
            'RMSE Train': rmse_train,
            'RMSE Test': rmse_test,
        }

# Вывести результаты
for key, value in results.items():
    print(f'Model: {key[0]}, Alpha: {key[1]}')
    print(
        f"R^2 (тренировочная выборка): {value['R2 Train']:.4f},"
        f" R^2 (тестовая выборка): {value['R2 Test']:.4f}, "
        f"RMSE (тренировочная выборка): {value['RMSE Train']:.4f},"
        f" RMSE (тестовая выборка): {value['RMSE Test']:.4f}\n")

# Построение графиков R2 и RMSE для каждой модели и альфа-комбинации
labels = [f"{model_name} (α={alpha})" for model_name in models.keys() for alpha in alphas]
r2_values = [results[(model_name, alpha)]['R2 Test'] for model_name in models.keys() for alpha in alphas]
rmse_values = [results[(model_name, alpha)]['RMSE Test'] for model_name in models.keys() for alpha in alphas]

x = np.arange(len(labels))

plt.figure(figsize=(20, 20))
fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('Модели и альфа-комбинации')
ax1.set_ylabel('R² Score', color=color)
ax1.bar(x - 0.2, r2_values, 0.4, label='R² Score', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('RMSE', color=color)
ax2.bar(x + 0.2, rmse_values, 0.4, label='RMSE', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Метрики')
plt.show()

models = ['Lasso', 'Ridge', 'ElasticNet']
r2_values = {model: [] for model in models}

for model in models:
    for alpha in alphas:
        r2_values[model].append(results[(model, alpha)]['R2 Test'])

plt.figure(figsize=(10, 6))

for model in models:
    plt.plot(alphas, r2_values[model], marker='o', label=model)

plt.title('Коэффициент детерминации (R2) при различных альфа')
plt.xlabel('Alpha значения')
plt.ylabel('R² Score')
plt.xticks(alphas)
plt.legend(title='Модели')
plt.grid()
plt.show()

'''Выводы:
1. Модель Lasso
    - Alpha = 0.01: Модель показывает наилучшие результаты среди Lasso, 
      с умеренной способностью объяснять вариации в данных.
    - Alpha = 0.3 и выше: R² значительно падает, достигая нуля и отрицательных значений. 
      RMSE увеличивается, что указывает на ухудшение качества предсказаний.
      Увеличение Alpha приводит к сильной регуляризации, что делает модель недостаточно гибкой для обучения на данных.  
2. Модель Ridge
    - Все значения Alpha (0.01, 0.3, 0.5, 0.8): R² остается стабильным для тренировочных и тестовых наборов. 
      RMSE также остается примерно на одном уровне. 
      Ridge показывает устойчивость к изменениям Alpha, 
      сохраняя приемлемое качество предсказаний при всех тестируемых значениях.
3. Модель ElasticNet
    - Alpha = 0.01: Наилучшие результаты среди ElasticNet, аналогично Ridge.
    - Alpha = 0.3 и выше: Снижение R² и увеличение RMSE, особенно при Alpha = 0.8.
      Как и в случае с Lasso, увеличение Alpha приводит к ухудшению качества модели.
  
Модель Lasso с Alpha = 0.01 показала лучшие результаты среди Lasso,
но все остальные значения Alpha значительно ухудшили качество.
Модели Ridge продемонстрировали стабильные результаты независимо от значения Alpha, 
что делает их надежными для данной задачи.
ElasticNet с Alpha = 0.01 также показал хорошие результаты, но при увеличении Alpha качество резко падает.'''

'''Задание №4. Настройка гиперпараметров модели'''

# Определение параметра alpha (лучший найденный ранее)
alpha_best = 0.01

# Список для хранения результатов
results = []

# Обучение модели с разным количеством итераций
for n_iter in [100, 300, 500, 800, 1000, 1500]:
    # Обучение модели SGDRegressor с методом градиентного спуска
    model = SGDRegressor(alpha=alpha_best, max_iter=n_iter, tol=1e-3)  # tol - критерий остановки
    model.fit(X_train, y_train)

    # Предсказания на тренировочной и тестовой выборках
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Вычисление метрик R² и RMSE
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    # Сохранение результатов
    results.append({
        'iterations': n_iter,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test
    })

# Вывод результатов
for result in results:
    print(f"Итерации: {result['iterations']}, "
          f"R² (тренировочная): {result['r2_train']:.4f}, "
          f"R² (тестовая): {result['r2_test']:.4f}, "
          f"RMSE (тренировочная): {result['rmse_train']:.4f}, "
          f"RMSE (тестовая): {result['rmse_test']:.4f}")


'''Выводы:

* R² (Коэффициент детерминации): для тренировочной и тестовой выборок колеблются в диапазоне от 0.1714 до 0.1819
  для тренировочной выборки и от 0.1760 до 0.1852 для тестовой. Наилучшие результаты наблюдаются при 800 итерациях,
  где R² достигает 0.1819 (тренировочная) и 0.1852 (тестовая). Это указывает на то, 
  что модель лучше всего объясняет вариацию данных на этих итерациях.
* RMSE (Корень из среднеквадратичной ошибки): варьируются от 0.9045 до 0.9103 
  для тренировочной выборки и от 0.9026 до 0.9077 для тестовой. 
  Наименьшие значения RMSE также были зафиксированы при 800 итерациях, что дополнительно подтверждает, 
  что именно на этом этапе модель демонстрирует наилучшие результаты.
* Сравнение по итерациям
    - Итерации 100 и 300: Эти итерации показывают относительно низкие значения R² и высокие значения RMSE, 
      что свидетельствует о недостаточной способности модели к обобщению.
    - Итерации 500 и 800: Здесь наблюдается улучшение как в R², 
      так и в RMSE, что говорит о том, что увеличение числа итераций положительно сказалось на качестве модели.
    - Итерации 1000 и 1500: Несмотря на некоторое снижение R² по сравнению с 800 итерациями, 
      значения остаются в пределах разумного, но RMSE не показывает значительных улучшений.

На основании анализа метрик можно сделать вывод,
что оптимальное количество итераций для данной модели составляет от 500 до 800. 
На этом этапе модель достигает лучших значений как по R², так и по RMSE, 
что указывает на её способность адекватно предсказывать целевую переменную.'''