import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Загрузите данные
data = pd.read_csv("../Hostel.csv")
data['Distance'] = data['Distance'].str.replace('km from city centre', '').astype(float)


features = data[
    ['Distance', 'summary.score', 'cleanliness', 'facilities', 'location.y', 'security', 'staff',
     'valueformoney']]

# Обработайте данные (пример: заполнение пропущенных значений медианой)

features = features.fillna(features.median())

# Примените t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

# Добавление кластеров к датафрейму
data['cluster'] = tsne_result.argmax(axis=1)  # Используем argmax для определения кластера

# Визуализация кластеров
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data['cluster'],
                      cmap='viridis', s=data['Distance'] ** 1.5)
plt.title('t-SNE Кластеризация хостелов')
plt.xlabel('t-SNE Component x')
plt.ylabel('t-SNE Component y')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.tight_layout()
plt.show()

# Вывод средних значений для каждого кластера
cluster_means1 = data.groupby('cluster').agg({'price.from': 'mean', 'Distance': 'mean', 'summary.score': 'mean',
                                              'cleanliness': 'mean'})
cluster_means2 = data.groupby('cluster').agg({'facilities': 'mean', 'location.y': 'mean',
                                              'security': 'mean', 'staff': 'mean', 'valueformoney': 'mean'})
print("Вывод средних значений для каждого кластера:")
print("\n", cluster_means1)
print("\n", cluster_means2)
