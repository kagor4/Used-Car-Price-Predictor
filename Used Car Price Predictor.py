#!/usr/bin/env python
# coding: utf-8

# ### Модель машинного обучения для определения рыночной стоимости автомобилей  
# 
# ## Описание проекта  
# 
# **Цель:** Разработка ML-модели для сервиса "Не бит, не крашен", которая быстро и точно предсказывает рыночную стоимость подержанных автомобилей на основе их характеристик.  
# 
# **Ключевые требования заказчика:**  
# - Высокая точность предсказания  
# - Оптимальное время обучения модели  
# - Быстрое время предсказания  
# 
# ## Данные  
# 
# **Источник:** Файл `/datasets/autos.csv`  
# 
# **Признаки:**  
# - Технические характеристики: `VehicleType`, `Gearbox`, `Power`, `Model`, `Kilometer`  
# - Исторические данные: `RegistrationYear`, `RegistrationMonth`, `Repaired`  
# - Прочие параметры: `FuelType`, `Brand`, `NumberOfPictures`  
# - Временные метки: `DateCrawled`, `DateCreated`, `LastSeen`  
# - Локация: `PostalCode`  
# 
# **Целевая переменная:**  
# - `Price` (цена в евро)  
# 
# ## Реализация  
# 
# **Этапы работы:**  
# 1. Предобработка данных:  
#    - Очистка от аномалий (нереальные значения мощности, года регистрации)  
#    - Работа с категориальными признаками (One-Hot Encoding, Target Encoding)  
#    - Feature engineering (возраст автомобиля, частота моделей)  
# 
# 2. Построение модели:  
#    - Тестирование различных алгоритмов (Random Forest, Gradient Boosting, LightGBM)  
#    - Оптимизация гиперпараметров  
#    - Подбор оптимального соотношения точности/скорости  
# 
# 3. Оценка качества:  
#    - Метрики: MAPE, RMSE, R²  
#    - Проверка на адекватность предсказаний  
# 
# **Стек технологий:**  
# - Python (Pandas, NumPy)  
# - Scikit-learn, LightGBM/XGBoost  
# - Matplotlib/Seaborn для визуализации  
# 
# ## Практическая ценность  
# 
# **Применение:**  
# - Пользователи приложения получают точную оценку стоимости автомобиля  
# - Увеличение конверсии сервиса за счет доверия к точности оценок  
# - Оптимизация работы платформы за счет быстрых предсказаний  

# ## Подготовка данных

# In[1]:


pip install tqdm


# In[2]:


pip install tqdm joblib


# In[3]:


get_ipython().system('pip install lightgbm')


# In[4]:


pip install --upgrade scikit-learn


# In[5]:


import pandas as pd
import numpy as np
import os
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from tqdm import tqdm
import time
from time import time
from joblib import parallel_backend
import lightgbm
print(lightgbm.__version__)


# In[6]:


# Определяем метрику RMSE для использования в кросс-валидации
def rmse_scorer(estimator, X, y):
    predictions = estimator.predict(X)
    return np.sqrt(mean_squared_error(y, predictions))

# Определяем k для кросс-валидации
k = 5


# In[7]:


# Настройка отображения для более широких столбцов
pd.set_option('display.max_colwidth', None)  # Устанавливаем неограниченную ширину столбцов
pd.set_option('display.max_rows', None)     # Устанавливаем неограниченное количество строк
pd.set_option('display.max_columns', None)  # Устанавливаем неограниченное количество столбцов


# In[8]:


data = '/datasets/autos.csv'

if os.path.exists(data):
    df = pd.read_csv(data)
else:
    print('Something is wrong')


# In[9]:


df.head(10)


# In[10]:


df.describe()


# In[11]:


df.info()


# Избавимся вот ненужных столбцов

# In[12]:


df = df.drop(['DateCrawled', 'RegistrationMonth', 'NumberOfPictures', 'PostalCode', 'LastSeen', 'DateCreated'], axis=1)


# Преобразуем название столбцов к нижнему регистру, а так же к змеиному регистру

# In[13]:


df.columns = df.columns.str.lower()
df = df.rename(columns={'vehicletype':'vehicle_type', 
                        'registrationyear':'registration_year', 
                        'fueltype':'fuel_type'})


# In[14]:


df.isna().sum()


# Обраотаем целевой признак price, пропусков данных нет, но есть цена равная нулю, восстановить по среднему было бы некоректно, так как признак целевой и это напрямую повлияет на прогноз.

# In[15]:


print("Колличество объявлений с нулевой ценой:",len(df.loc[df['price'] == 0]))


# In[16]:


df = df.loc[df['price'] != 0]


# Имеются пропуски в model к сожалению их нельзя восстановит по brand придется удалить

# In[17]:


print("Колличество объявлений с пропущенной моделью:", len(df.loc[df['model'].isna()]))


# In[18]:


df = df.loc[~df['model'].isna()]


# In[19]:


df['brand'].unique()


# Колличество пропусков в vehicle_type около 10 процентов, данных слишком много чтобы просто от них избавится, если заменить на среднюю это тоже может сказаться на точности предсказания. Если брать в расчет что в дальнейшем пользователи при оценке автомобиля могут так же не вводить тип кузова, то стоит заменить пропущенные значения на unknown

# In[20]:


print("Колличество объявлений с незаполненным типом кузова:", len(df.loc[df['vehicle_type'].isna()]))


# In[21]:


df['vehicle_type'] = df['vehicle_type'].fillna('unknown')


# Количество явных некорректных данных registration_year незначительно, можно их удалить

# In[22]:


print("Колличество объявлений с некорректной годом:",len(df.loc[(df['registration_year'] > 2021) | (df['registration_year'] < 1769)]))


# In[23]:


export_date = datetime.strptime('2024-06-05', '%Y-%m-%d')
df = df.loc[(df['registration_year'] >= 1900) & (df['registration_year'] <= export_date.year)]


# Колличество пропусков в gearbox тоже велико. Заменю на наиболее встречающийся тип коробки в модели.

# In[24]:


print("Колличество объявлений с незаполненным типом коробки:", len(df.loc[df['gearbox'].isna()]))


# In[25]:


df['gearbox'] = df['gearbox'].fillna(df
                                         .groupby('model')['gearbox']
                                         .transform(lambda x: x.value_counts().idxmax())
                                        )


# Имеются значения power равные 0 и больше 1000 что являеться некорректным, можно заменить их на медиану по модели

# In[26]:


print("Колличество объявлений с некорректной мощностью:",len(df.loc[(df['power'] > 1000) | (df['power'] <= 0)]))


# In[27]:


df.loc[(df['power'] > 1000) | (df['power'] <= 0), 'power'] = None
df['power'] = df['power'].fillna(df.groupby('model')['power'].transform('median'))
df = df.loc[~df['power'].isna()]
df['power'] = df['power'].astype('int64')


# Пропуски в fuel_type заменю на среднее по моделям

# In[28]:


df['fuel_type'] = df['fuel_type'].fillna(df.groupby('model')['fuel_type'].transform(lambda x: x.value_counts().idxmax()))


# Пропуски в repaired состоявляют треть от данных. Замению на 'unknown'

# In[29]:


df['repaired'].fillna('unknown', inplace=True)


# Признак date_created преобразуем в количество дней с момента 2014-03-01.

# In[32]:


df = df.drop_duplicates()


# In[33]:


df.info()


# In[34]:


df.isna().sum()


# In[35]:


df.describe()


# In[36]:


df_reg = df.copy()


# В этом разделе мы произвели предобработку данных. На входе мы получили таблицу с более чем 350 тыс. строк и 16 столбцами. Мы перевели названия столбцов в более удобочитаемый нижний и змеиный регистр, удалили неинформативные столбцы, заполнили пропуски, избавились от аномалий. Удалили дубликаты и перевели категориальные значения столбцов в количественные. В итоге у нас образовалась таблица в 310 тыс. строк.

# ## Обучение моделей

# Разделим выборки на обучающую и тестовую

# In[37]:


# Разделение на целевую переменную и признаки
target = df['price']
features = df.drop('price', axis=1)

# Разделение на тренировочную и тестовую выборки
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=12345)

print(features_train.shape)
print(features_test.shape)


# In[38]:


# Определение категориальных и числовых признаков
categorical_features = features_train.select_dtypes(include=['object', 'category']).columns
numerical_features = features_train.select_dtypes(exclude=['object', 'category']).columns

# Проверка числовых признаков на наличие нечисловых данных
for col in numerical_features:
    if not pd.api.types.is_numeric_dtype(features_train[col]):
        print(f"Столбец {col} содержит нечисловые данные: {features_train[col].unique()}")

# Преобразование всех значений в числовых столбцах в числовой формат
for col in numerical_features:
    features_train[col] = pd.to_numeric(features_train[col], errors='coerce')
    features_test[col] = pd.to_numeric(features_test[col], errors='coerce')

# Масштабирование числовых признаков
scaler = StandardScaler()
features_train[numerical_features] = scaler.fit_transform(features_train[numerical_features])
features_test[numerical_features] = scaler.transform(features_test[numerical_features])


# In[39]:


# Кодирование категориальных признаков
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')


# In[40]:


encoded_train = encoder.fit_transform(features_train[categorical_features])
encoded_test = encoder.transform(features_test[categorical_features])

# Объединение числовых и закодированных категориальных признаков
features_train_processed = np.hstack([features_train[numerical_features].values, encoded_train])
features_test_processed = np.hstack([features_test[numerical_features].values, encoded_test])

target_train_processed = target_train.values
target_test_processed = target_test.values


# In[41]:


print(features_train_processed.shape)
print(features_test_processed.shape)


# # 2.1 Модель линейной регрессии

# In[42]:


model = LinearRegression()

# Оценка модели с использованием кросс-валидации
scores = cross_val_score(model, features_train_processed, target_train_processed, cv=5, scoring='neg_mean_squared_error')
rmse_scores_lr_cv = (-scores)**0.5
mean_rmse_scores_lr_cv = round(rmse_scores_lr_cv.mean(), 2)
print(f'RMSE на кросс-валидации: {rmse_scores_lr_cv}')
print(f'Среднее RMSE на кросс-валидации: {mean_rmse_scores_lr_cv}')


# In[43]:


# # Обучение модели на полной тренировочной выборке и тестирование на тестовой выборке
# model.fit(features_train_processed, target_train)
# predictions_test_processed = model.predict(features_test_processed)
# test_rmse = mean_squared_error(target_test, predictions_test_processed)**0.5
# print(f'RMSE на тестовой выборке: {test_rmse}')


# # 2.2 Модель дерева решений

# In[44]:


#разделим выборки на обучающую и тестовую
target_oe = df['price']
features_oe = df.drop('price', axis=1)
features_train_oe, features_test_oe, target_train_oe, target_test_oe = train_test_split(features_oe, target_oe, test_size=0.25, random_state=12345) 


print(features_train_oe.shape)
print(features_test_oe.shape)


# In[45]:


cols = ['vehicle_type','registration_year', 'gearbox', 'power', 
        'model', 'kilometer','fuel_type', 'brand', 'repaired']

# Создаем экземпляр кодировщика
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Кодируем категориальные признаки
features_train_encoded = features_train_oe.copy()
features_train_encoded[cols] = encoder.fit_transform(features_train_oe[cols])


# In[46]:


param_grid = {
    'max_depth': range(3, 20, 3),
    # Другие гиперпараметры, если нужно
}

# Создаем экземпляр модели
model = DecisionTreeRegressor(random_state=12345)

# Создаем экземпляр объекта GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Обучаем GridSearchCV на обучающих данных
grid_search.fit(features_train_encoded, target_train_oe)

# Извлекаем результаты и рассчитываем RMSE
best_params = grid_search.best_params_
best_neg_mse = grid_search.best_score_
rmse_scores_dt_cv = np.sqrt(-best_neg_mse)

# Выводим лучшие параметры и результаты на обучающей выборке
print("Лучшие гиперпараметры:", best_params)
print("Лучший RMSE для модели на обучающей выборке (кросс-валидация):", rmse_scores_dt_cv)


# # 2.3 Модель случайного леса

# In[47]:


# Создаем ColumnTransformer для преобразования категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ],
    remainder='passthrough'  # Оставляем остальные признаки без изменений
)


# In[48]:


# Создаем экземпляр модели
model = RandomForestRegressor(random_state=12345)

# Создаем Pipeline для предварительной обработки и моделирования
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])


# In[49]:


import time

# Настройка TQDM для параллельных задач
class TqdmParallel(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining}')
        super().__init__(*args, **kwargs)

    def __call__(self, func):
        def wrap(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrap

# Определяем сетку гиперпараметров
param_grid = {
    'model__n_estimators': [10, 50, 100],
    'model__max_depth': range(5, 20, 5),
}

# Создаем экземпляр объекта RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    scoring='neg_mean_squared_error',
    n_iter=5,
    cv=5,
    random_state=12345,
    n_jobs=-1  # Используем все процессоры
)

# Начало замера времени
start_time = time.time()

# Обучаем RandomizedSearchCV на обучающих данных с прогресс-баром
with parallel_backend('loky', n_jobs=-1):
    with TqdmParallel(total=random_search.n_iter * random_search.cv) as progress_bar:
        random_search.fit(features_train_oe, target_train_oe)

# Конец замера времени
end_time = time.time()

# Выводим лучшие параметры
print("Лучшие гиперпараметры:", random_search.best_params_)
print("Время выполнения: {:.2f} секунд".format(end_time - start_time))


# In[50]:


# %%time

# # Оцениваем модель с использованием кросс-валидации
# cv_scores = cross_val_score(random_search.best_estimator_, features_train_oe, target_train_oe, cv=5, scoring='neg_mean_squared_error')


# In[51]:


# Получаем лучший score
best_score = random_search.best_score_

# Преобразуем best_score из neg_mean_squared_error в RMSE
rmse_scores_rf_cv = (-best_score) ** 0.5
print("Лучшее RMSE на кросс-валидации:", rmse_scores_rf_cv)

# Сохраняем результаты в DataFrame
results_df = pd.DataFrame(random_search.cv_results_)

# Выбираем и сортируем интересующие нас колонки
results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
results_df['mean_test_score'] = (-results_df['mean_test_score']) ** 0.5  # Преобразуем в RMSE

# Сортируем по возрастанию RMSE
results_df = results_df.sort_values(by='mean_test_score')

# Выводим топ-5 результатов
print("Топ-5 результатов:")
results_df.head()


# Среднее значение RMSE на кросс-валидации составило 1690.57. Наилучший результат был достигнут при параметрах {'model__n_estimators': 100, 'model__max_depth': 15}, что также соответствует минимальной среднеквадратической ошибке в 1690.57. Однако, даже при этом параметре наблюдается значительная вариативность ошибки (стандартное отклонение ≈ 38382.09). Топ-5 лучших моделей показывают, что увеличение глубины дерева и количества деревьев не всегда приводит к улучшению результата, поскольку более сложные модели, как правило, имеют более высокие значения RMSE.

# # 2.4 LightGBM

# In[52]:


# выделим признаки и целевой признак
target = df['price']
features = df.drop('price', axis=1)

# переведем категориальные признаки в тип category, который требуется для LightGBM
for c in features.columns:
  col_type = features[c].dtype
  if col_type == 'object':
    features[c] = features[c].astype('category')

#разделим выборки на обучающую и тестовую
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=12345) 

# проверим размер выборок
print(features_train.shape)
print(features_test.shape)


# In[53]:


# построим модель без подбора гиперпараметров
model = lgb.LGBMRegressor(random_state=12345)


# In[54]:


# cv_scores = cross_val_score(model, features_train, target_train, scoring='neg_mean_squared_error', cv=5)

# rmse_cv = np.sqrt(-cv_scores)

# rmse_scores_lgbm_cv = np.mean(rmse_cv)
# print("Среднее RMSE по кросс-валидации:", rmse_scores_lgbm_cv)


# Найдем наилучшие гиперпараметры для LightGBM

# In[55]:


param_grid = {'n_estimators': [100, 500, 1000], 'num_leaves': [n for n in range(20, 300, 20)]}


# In[56]:


# model = lgb.LGBMRegressor(random_state=12345)

# tuning_model=GridSearchCV(estimator=model,
#                           param_grid=param_grid,
#                           scoring='neg_root_mean_squared_error',
#                           cv=3,
#                           verbose=3)

# tuning_model.fit(features_train, target_train)
# display(tuning_model.best_params_)
# display(tuning_model.best_score_*-1)


# In[57]:


# cv_scores = cross_val_score(model, features, target, scoring='neg_mean_squared_error', cv=5)

# rmse_cv = np.sqrt(-cv_scores)

# rmse_scores_lgbm_cv = np.mean(rmse_cv)
# print("Среднее RMSE по кросс-валидации:", rmse_scores_lgbm_cv)


# In[58]:


rmse_scores_lgbm_cv = 1687.18


# Закомментил код, так как очень долго он выполняется, лучший результат получился:
# {'n_estimators': 500, 'num_leaves': 80}
# 1687.1815711334286

# В этом разделе мы обучили разные модели, подобрали гиперпараметры для них и обнаружили, что модель LightGBM с гиперпараметрами 'n_estimators': 500, 'num_leaves': 80 дает наименьшее значение rmse: 1687.18

# ## Анализ моделей

# # 3.1 Модель линейной регрессии

# In[59]:


model = LinearRegression()

start_cpu_time = time.process_time()
start_wall_time = time.time()

model.fit(features_train_processed, target_train_processed)

end_cpu_time = time.process_time()
end_wall_time = time.time()

cpu_time_train_lr = end_cpu_time - start_cpu_time
wall_time_train_lr = end_wall_time - start_wall_time

print(f"CPU time для обучения модели: {cpu_time_train_lr:.2f} секунд")
print(f"Wall time для обучения модели: {wall_time_train_lr:.2f} секунд")


# In[60]:


start_cpu_time_pred = time.process_time()
start_wall_time_pred = time.time()

predictions_train = model.predict(features_train_processed)

end_cpu_time_pred = time.process_time()
end_wall_time_pred = time.time()

cpu_time_pred_lr = end_cpu_time_pred - start_cpu_time_pred
wall_time_pred_lr = end_wall_time_pred - start_wall_time_pred

print(f"CPU time для выполнения предсказаний: {cpu_time_pred_lr:.2f} секунд")
print(f"Wall time для выполнения предсказаний: {wall_time_pred_lr:.2f} секунд")


# # 3.2 Модель дерева решений

# In[61]:


model_dt = DecisionTreeRegressor(random_state=12345, max_depth=12)

start_cpu_time_train_dt = time.process_time()
start_wall_time_train_dt = time.time()

model_dt.fit(features_train_processed, target_train_processed)

end_cpu_time_train_dt = time.process_time()
end_wall_time_train_dt = time.time()

cpu_time_train_dt = end_cpu_time_train_dt - start_cpu_time_train_dt
wall_time_train_dt = end_wall_time_train_dt - start_wall_time_train_dt

print(f"CPU time для обучения модели решающего дерева: {cpu_time_train_dt:.2f} секунд")
print(f"Wall time для обучения модели решающего дерева: {wall_time_train_dt:.2f} секунд")


# In[62]:


start_cpu_time_pred_dt = time.process_time()
start_wall_time_pred_dt = time.time()

predictions_train_dt = model_dt.predict(features_train_processed)

end_cpu_time_pred_dt = time.process_time()
end_wall_time_pred_dt = time.time()

cpu_time_pred_dt = end_cpu_time_pred_dt - start_cpu_time_pred_dt
wall_time_pred_dt = end_wall_time_pred_dt - start_wall_time_pred_dt

print(f"CPU time для выполнения предсказаний решающего дерева: {cpu_time_pred_dt:.2f} секунд")
print(f"Wall time для выполнения предсказаний решающего дерева: {wall_time_pred_dt:.2f} секунд")


# # 3.3 Модель случайного леса

# In[63]:


model_rf = RandomForestRegressor(random_state=12345, max_depth=15, n_estimators=100)

start_cpu_time_train_rf = time.process_time()
start_wall_time_train_rf = time.time()

model_rf.fit(features_train_processed, target_train_processed)

end_cpu_time_train_rf = time.process_time()
end_wall_time_train_rf = time.time()

cpu_time_train_rf = end_cpu_time_train_rf - start_cpu_time_train_rf
wall_time_train_rf = end_wall_time_train_rf - start_wall_time_train_rf

print(f"CPU time для обучения модели случайного леса: {cpu_time_train_rf:.2f} секунд")
print(f"Wall time для обучения модели случайного леса: {wall_time_train_rf:.2f} секунд")


# In[64]:


start_cpu_time_pred_rf = time.process_time()
start_wall_time_pred_rf = time.time()

predictions_train_rf = model_rf.predict(features_train_processed)

end_cpu_time_pred_rf = time.process_time()
end_wall_time_pred_rf = time.time()

cpu_time_pred_rf = end_cpu_time_pred_rf - start_cpu_time_pred_rf
wall_time_pred_rf = end_wall_time_pred_rf - start_wall_time_pred_rf

print(f"CPU time для выполнения предсказаний случайного леса: {cpu_time_pred_rf:.2f} секунд")
print(f"Wall time для выполнения предсказаний случайного леса: {wall_time_pred_rf:.2f} секунд")


# # 3.4 Модель LightGBM

# In[65]:


model_lgb = lgb.LGBMRegressor(random_state=12345, n_estimators=500, num_leaves=80)

start_cpu_time_train_lgb = time.process_time()
start_wall_time_train_lgb = time.time()

model_lgb.fit(features_train, target_train)

end_cpu_time_train_lgb = time.process_time()
end_wall_time_train_lgb = time.time()

cpu_time_train_lgb = end_cpu_time_train_lgb - start_cpu_time_train_lgb
wall_time_train_lgb = end_wall_time_train_lgb - start_wall_time_train_lgb

print(f"CPU time для обучения модели LightGBM: {cpu_time_train_lgb:.4f} секунд")
print(f"Wall time для обучения модели LightGBM: {wall_time_train_lgb:.4f} секунд")


# In[66]:


start_cpu_time_pred_lgb = time.process_time()
start_wall_time_pred_lgb = time.time()

predictions_train_lgb = model_lgb.predict(features_train)

end_cpu_time_pred_lgb = time.process_time()
end_wall_time_pred_lgb = time.time()

cpu_time_pred_lgb = end_cpu_time_pred_lgb - start_cpu_time_pred_lgb
wall_time_pred_lgb = end_wall_time_pred_lgb - start_wall_time_pred_lgb

print(f"CPU time для выполнения предсказаний LightGBM: {cpu_time_pred_lgb:.2f} секунд")
print(f"Wall time для выполнения предсказаний LightGBM: {wall_time_pred_lgb:.2f} секунд")


# In[67]:


tabledata = {
    "": [
        "линейная регрессия: обучение", 
        "линейная регрессия: предсказание",
        "решающее дерево: обучение", 
        "решающее дерево: предсказание",
        "случайный лес: обучение", 
        "случайный лес: предсказание",
        "LightGBM: обучение", 
        "LightGBM: предсказание"
    ],
    "CPU-times": [
        cpu_time_train_lr, cpu_time_pred_lr,
        cpu_time_train_dt, cpu_time_pred_dt,
        cpu_time_train_rf, cpu_time_pred_rf,
        cpu_time_train_lgb, cpu_time_pred_lgb
    ],
    "Wall time": [
        wall_time_train_lr, wall_time_pred_lr,
        wall_time_train_dt, wall_time_pred_dt,
        wall_time_train_rf, wall_time_pred_rf,
        wall_time_train_lgb, wall_time_pred_lgb
    ],
    "RMSE на кросс-валидации": [
        mean_rmse_scores_lr_cv, mean_rmse_scores_lr_cv,  # одно и то же значение для обучения и предсказания
        rmse_scores_dt_cv, rmse_scores_dt_cv,
        rmse_scores_rf_cv, rmse_scores_rf_cv,
        rmse_scores_lgbm_cv, rmse_scores_lgbm_cv
    ]
}

df = pd.DataFrame(tabledata)

print(df)


# Рекомендация по моделям:
# 
# - Лучшее качество (наименьшее RMSE): LightGBM (1687.18).
# - Обучение: Решающее дерево (5.37 сек).
# - Предсказание: Решающее дерево (0.18 сек).
# 
# Если приоритет — качество модели, выбираем LightGBM. Если важнее время выполнения, оптимальным выбором будет Решающее дерево. LightGBM имеет самое долгое время предсказания. 
# 
# Заказчику важны показатели:
# 
# - качество предсказания;
# - время обучения модели;
# - время предсказания модели.
# 
# Модель случайного леса показывает наилучшую скорость. RMSE лучшее у модели LightGBM. LightGBM модель обучается и предсказывает приемлемо по времени, 39.42 сек и 14.78 соответственно. Я рекомендую использовать именно эту модель.

# In[68]:


# predictions_test = model_rf.predict(features_test_processed)
# rmse = mean_squared_error(target_test_processed, predictions_test)**0.5
# print(rmse)


# In[69]:


predictions_test = model_lgb.predict(features_test)
rmse = mean_squared_error(target_test, predictions_test)**0.5
print(rmse)


# In[70]:


# посмотрим на то, какие признаки влияют больше всего
lgb.plot_importance(model_lgb, height=.5);


# Основные признаки, влияющие на цену автомобиля, включают мощность двигателя и год регистрации автомобиля. Среднее влияние оказывают такие признаки, как пробег и модель автомобиля. Наименьшее влияние на цену имеют: был ли автомобиль в ремонте, тип кузова, бренд, тип коробки передач и тип топлива.

# In[71]:


x_ax = range(len(target_test))

plt.figure(figsize=(12, 6))
plt.scatter(target_test, predictions_test, alpha=0.5)
plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], color='red', linestyle='--', label='Идеальная линия')
plt.title("Определение стоимости автомобилей: тестовые vs предсказанные данные")
plt.xlabel('Фактические значения (цены)')
plt.ylabel('Предсказанные значения (цены)')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# На диаграмме видно, что предсказанные значения имеют тенденцию следовать за фактическими, однако наблюдается значительное рассеяние точек, особенно при низких значениях цены. Это говорит о том, что модель LightGBM неплохо справляется с предсказанием стоимости автомобилей. Точки выше линии означают переоценку цен, а ниже линии — недооценку. Основная масса точек находится в средней части графика, что подтверждает приемлемое качество модели.

# Быстрее всего учится модель случайного леса: 5 мин 37 сек. Модель LightGBM имеет приемлемую скорость обучения и предсказания (в пределах 40 сек.) и при этом самую низкую RMSE из всех моделей: на тестовой выборке результат 1611.68

# # Общий вывод
# 
# 1. На входе мы получили таблицу с более чем 350 тыс. строк и 16 столбцами. Мы перевели названия столбцов в более удобочитаемый нижний и змеиный регистр, удалили неинформативные столбцы, заполнили пропуски, избавились от аномалий. Удалили дубликаты.
# 
# 
# 2. Для модели линейной регрессии перевели категориальные значения столбцов в количественные. В итоге у нас образовалась таблица в 310 тыс. строк. Для моделей решающего дерева и случайного леса применили порядковое кодирование, в итоге для них количесвто столбцов не изменилось, а подбор гиперпараметров стал быстрее. Мы обучили разные модели, подобрали гиперпараметры для них и обнаружили, что модель LightGBM с гиперпараметрами 'n_estimators': 500, 'num_leaves': 80 дает наименьшее значение RMSE: 1611.68.
# 
# 
# 3. Мы проанализировали все модели и обнаружили, что дольше всего учится модель случайного леса: 5 мин 37 сек. Модель LightGBM имеет приемлемую скорость обучения (в пределах 40 сек.). Время предсказания у модели LightGBM самое большое - 14 секунд, но при этом она имеет самую низкую RMSE из всех моделей: на валидационной выборке результат 1687.18. На тестовой выборке - 1611.68
# 
# Исходя из требований заказчика по скорости обучения, придсказания и качества модели, рекомендуем модель LightGBM, поскольку она имеет самые оптимальные характеристики.
