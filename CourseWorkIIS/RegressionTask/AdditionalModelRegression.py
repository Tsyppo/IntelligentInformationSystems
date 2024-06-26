import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Загрузка данных
data = pd.read_csv("../Hostel.csv")

features = ['price.from', 'Distance', 'atmosphere', 'cleanliness', 'facilities']

# Очистка данных и заполнение пропущенных значений
data['Distance'] = data['Distance'].str.replace('km from city centre', '').astype(float)
data['summary.score'] = data['summary.score'].fillna(data['summary.score'].mean())
target = 'summary.score'
data[features] = data[features].fillna(data[features].mean())

# Разделение данных
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = train_data[features]
y_train = train_data[target]

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Проверка работы модели
X_test = test_data[features]
y_test = test_data[target]

predictions = model.predict(X_test)

# Точность на основе средней квадратной ошибке (MSE)
mse = mean_squared_error(y_test, predictions)
accuracy = 1 - mse / y_test.var()
print(f"\nТочность модели: {accuracy * 100:.2f}%")

importances = model.feature_importances_
normalized_importances = importances / importances.sum()

# Выбор трёх наиболее важных признаков
top_importances = normalized_importances.argsort()[-3:][::-1]
print("\nТри наиболее важных признака:")
for idx in top_importances:
    print(f"{features[idx]}: {normalized_importances[idx]}")