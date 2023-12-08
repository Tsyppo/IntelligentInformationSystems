import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

# Загрузите данные
data = pd.read_csv("../Hostel.csv")
data['Distance'] = data['Distance'].str.replace('km from city centre', '').astype(float)


features = data[['Distance', 'summary.score', 'cleanliness', 'facilities', 'location.y', 'security', 'staff',
                 'valueformoney']]

# Обработайте данные
features = features.fillna(features.median())
data['summary.score'] = data['summary.score'].fillna(data['summary.score'].mean())

# Масштабирование признаков
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Параметры SOM
grid_size = 10  # Размер сетки
n_iterations = 1000  # Количество итераций обучения

# Создание SOM
som = MiniSom(grid_size, grid_size, scaled_features.shape[1], sigma=1.0, learning_rate=0.5)

# Инициализация весов SOM
som.random_weights_init(scaled_features)

# Обучение SOM
som.train_random(scaled_features, n_iterations)

# Получение карты активации (BMU - Best Matching Unit) для каждого образца
activation_map = som.activation_response(scaled_features)

# Определение кластера для каждого образца
clusters = np.zeros(len(scaled_features))

for i, sample in enumerate(scaled_features):
    clusters[i] = np.argmin([np.linalg.norm(sample - weight) for weight in som.get_weights().reshape(-1, scaled_features.shape[1])])

# Визуализация карты активации
plt.imshow(activation_map, cmap='Blues', interpolation='none')
plt.colorbar()
plt.title('SOM Activation Map')
plt.show()

# Добавление кластеров к датафрейму
data['cluster'] = clusters

# Рассчитать средние значения для каждого признака по всем кластерам
overall_means = data.groupby('cluster')[['price.from', 'Distance', 'summary.score', 'cleanliness', 'facilities',
                                         'location.y', 'security', 'staff', 'valueformoney']].mean()

# Рассчитать дисперсию для каждого признака
feature_variances = features.var()

# Найти топ-3 признака с наибольшей дисперсией
top_features = feature_variances.nlargest(3).index

# Вывести топ-3 признака с наибольшей дисперсией
print("Top 3 Most Influential Features:")
print(top_features)


# Найти топ-3 кластера по количеству нейронов
top_clusters = data['cluster'].value_counts().nlargest(3).index

# Вывести информацию и средние значения для каждого из топ-3 кластеров
for cluster in top_clusters:
    cluster_data = data[data['cluster'] == cluster]

    print(f"\nCluster {cluster}:")
    print(cluster_data[['hostel.name', 'City', 'price.from', 'summary.score']])

    # Вывод средних значений для каждого кластера
    cluster_means = cluster_data[['price.from', 'Distance', 'summary.score', 'cleanliness', 'facilities',
                                  'location.y', 'security', 'staff', 'valueformoney']].mean()

    print("\nAverage Values for Cluster:")
    print(cluster_means)
    print("=" * 50)

