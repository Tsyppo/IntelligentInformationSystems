import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv("../Hostel.csv")

# Заполнение пропущенных значений
data['Distance'] = data['Distance'].str.replace('km from city centre', '').astype(float)
data['summary.score'] = data['summary.score'].fillna(data['summary.score'].mean())

# Выбор нужных признаков
features = ['price.from', 'Distance', 'atmosphere', 'cleanliness', 'facilities']
target = 'summary.score'

# Категоризация рейтинга
data['rating_category'] = pd.cut(data[target], bins=[0, 2, 4, 6, 7, 8, 9, 10],
                                 labels=['Very Low', 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Very High'])

# Разделение данных на обучающий и тестовый наборы
X = data[features]
y = data['rating_category']

# Заполнение пропущенных значений в X
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных (стандартизация)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Использование DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе
y_pred = classifier.predict(X_test_scaled)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)

# Вывод точности модели
print(f"\nТочность модели: {accuracy * 100:.2f}%")

# Вывод трех наиболее важных признаков
importances = classifier.feature_importances_
indices = (-importances).argsort()[:3]

print("Наиболее важные признаки:")
for f in range(3):
    print(f"{features[indices[f]]}: {importances[indices[f]]}")