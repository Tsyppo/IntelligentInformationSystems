import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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


# Применение функции define_group для создания целевой переменной
data['group'] = data.apply(define_group, axis=1)

# Преобразование категориальных признаков в числовые
le_city = LabelEncoder()
le_rating_band = LabelEncoder()

data['City'] = le_city.fit_transform(data['City'])
data['rating.band'] = le_rating_band.fit_transform(data['rating.band'])

# Разделение данных на обучающую и тестовую выборки
X = data[['City', 'rating.band', 'price.from']]
y = data['group']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = clf.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Точность модели: {accuracy * 100:.2f}%")

# Важность признаков
feature_importance = clf.feature_importances_
print("Feature Importance:")
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")
