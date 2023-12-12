import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Загрузим данные
data = pd.read_csv("../Hostel.csv")

# Выбранные города
selected_cities = ['Kyoto', 'Osaka', 'Hiroshima', 'Tokyo', 'Fukuoka-City']

# Выбранные рейтинговые группы
selected_rating_groups = ['Rating', 'Superb', 'Very Good', 'Fabulous', 'Good']


# Определим условия для формирования групп
def define_group(row):
    if row['City'] in selected_cities and row['rating.band'] in selected_rating_groups:
        if 1000 <= row['price.from'] <= 2000:
            return 'Group ' + row['City'] + ' ' + row['rating.band'] + ' Low'
        elif 2001 <= row['price.from'] <= 3000:
            return 'Group ' + row['City'] + ' ' + row['rating.band'] + ' Medium'
        elif row['price.from'] >= 3001:
            return 'Group ' + row['City'] + ' ' + row['rating.band'] + ' High'
    return 'Other'

# Создадим столбец 'Group'
data['Group'] = data.apply(define_group, axis=1)

# Исключим столбец 'hostel.name' из анализа числовых значений
numerical_columns = ['price.from']
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Заполним отсутствующие значения в числовых столбцах медианными значениями
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

# Закодируем категориальные переменные (City и rating.band)
label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])
data['rating.band'] = label_encoder.fit_transform(data['rating.band'])
data['Group'] = label_encoder.fit_transform(data['Group'])

# Разбиваем данные на тренировочный и тестовый наборы
X = data[['price.from', 'City', 'rating.band']]
y = data['Group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Обучаем модель
model.fit(X_train, y_train)

# Предсказываем значения для тестового набора
y_pred = model.predict(X_test)

# Оцениваем точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Выведем важность признаков
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
print("\n", feature_importances)



