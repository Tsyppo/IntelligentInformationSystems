import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
df = pd.read_csv("../Hostel.csv")
df = df.fillna(0)

# Выбор нужных признаков
features = ['summary.score', 'atmosphere', 'cleanliness', 'facilities', 'staff', 'valueformoney']
target = 'price.from'

# Разделение данных на обучающий и тестовый наборы
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Масштабирование данных (стандартизация)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе
y_pred_rf = rf_regressor.predict(X_test_scaled)

# Оценка качества модели RandomForestRegressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
accuracy_rf = 1 - mse_rf / y_test.var()

# Вывод результатов
print("Тестируемые строки:")
print(X_test)

# Вывод точности модели RandomForestRegressor
print(f"\nТочность модели RandomForestRegressor: {accuracy_rf * 100:.2f}%")
importances = rf_regressor.feature_importances_
normalized_importances = importances / importances.sum()

# Выбор трёх наиболее важных признаков
top_importances = normalized_importances.argsort()[-3:][::-1]
print("\nТри наиболее важных признака:")
for idx in top_importances:
    print(f"{features[idx]}: {normalized_importances[idx]}")