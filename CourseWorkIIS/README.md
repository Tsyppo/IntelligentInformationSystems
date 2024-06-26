# Курсовая работа по предмету Интеллектуальные информационные системы

В ходе исследования был проанализирован набор данных о хостелах Японии. Были выявлены методы регрессии, позволяющие 
прогнозировать рейтинг хостела, кластеризации помогающие сгруппировать хостелы по разным признакам и классификации, 
помогающие предсказывать к какой группе относится хостел. На основе методов были выявлены признаки, влияющие на оценку
рейтинга хостела, по каким признакам группируются лучшие хостелы и какие признаки хостела лучше всего подходят при 
разбиении на группы.

## Решение задачи регрессии.

### Постановка задачи:

>Предсказание рейтинга хостела на основе ключевых признаков цены, атмосферы, расстоянии от центра города, чистоте, удобства.

### Вывод

Можем подвести итог по решению задачи регрессии. Lasso-регрессия предоставляет более высокую точность, что может быть 
ключевым фактором в выборе модели для данной задачи предсказания рейтинга хостела. Среди наиболее важных признаков обе 
модели выделили три одинаковых показателя атмосфера, чистота и удобство. Данные методы полезны для тех, кто собирается 
открыть свой хостел, чтобы попытаться предсказать рейтинг своего будущего хостела основываясь, а также выяснить какие 
показатели являются более важными и на них сделать упор.

## Решение задачи кластеризации.

### Постановка задачи:

> Сегментируем данные по группам для выявления ключевых признаков, по которым группируются лучшие хостелы. Кластеризация 
> хостелов будет проводиться на основе признаков: расстояния от центра города и рейтинга, по общей оценке, удобства, 
> чистоты, локации, безопасности, персонала, атмосферы, цены денег.

### Графики

Полученный график t-SNE:

![Графики моделей](ClusteringTask/img/MainModelClustering.png)

Самым интересным и результирующим признаком оказался Distance, потому что только этот параметр наиболее чётко показывает 
разбиение на два кластера.

Пример построенного графика Self-Organizing Map:

![Графики моделей](ClusteringTask/img/AdditionModelClustering.png)

Чем более тёмный кластер, тем большая группа образовалась на этом месте. Всего было выделено 100 кластеров (0-99) И из 
них можно выделить 3 основных: 

![Кластеры моделей](ClusteringTask/img/Cluster1.png)

![Кластеры моделей](ClusteringTask/img/Cluster2.png)

![Кластеры моделей](ClusteringTask/img/Cluster3.png)

Вот три самых результирующих признака:

![Топ 3 признака](ClusteringTask/img/Top3.png)

### Вывод

Сравнив два метода, можно сделать вывод, что оба метода подходят для выделения групп. В случае первого метода группы 
были выделены в основном из-за признака основанном на дистанции от центра города. Во втором методе по трём признакам 
дистанция, удобство и чистота. В обоих случая дистанция была ключевым признаком для разделения данных на группы. Данные 
методы очень полезны в случае, если вы хотите открыть свой хостел, и вам нужно выбрать место. При выводе дополнительных 
столбцов эти методы могут показать, где и на каком расстоянии находятся большинство хостелов, опираясь так же на их рейтинг.

## Решение задачи классификации.

### Постановка задачи:

>Решим задачу разделением хостелов на группы в зависимости от признаков: ценового диапазона, по городам и рейтинговой 
>группы. Выявить какие признаки оказывают наиболее большое влияние на результат.

### Вывод

Сравнив два метода, можно сделать вывод, что оба метода подходят для выделения групп. В случае первого метода группы 
были выделены в основном из-за признака основанном на дистанции от центра города. Во втором методе по трём признакам 
дистанция, удобство и чистота. В обоих случая дистанция была ключевым признаком для разделения данных на группы. Данные 
методы очень полезны в случае, если вы хотите открыть свой хостел, и вам нужно выбрать место. При выводе дополнительных 
столбцов эти

## Заключение

В ходе исследования был проанализирован набор данных о хостелах Японии. Были выявлены методы регрессии, позволяющие 
прогнозировать рейтинг хостела, кластеризации помогающие сгруппировать хостелы по разным признакам и классификации, 
помогающие предсказывать к какой группе относится хостел. На основе методов были выявлены признаки, влияющие на оценку 
рейтинга хостела, по каким признакам группируются лучшие хостелы и какие признаки хостела лучше всего подходят при 
разбиении на группы.

Решили задачу регрессии через методы Лассо-регрессии и DecisionTreeRegressor. 
Точность Лассо-регрессии составляет - 90.46%, точность DecisionTreeRegressor - 57.40%. 
Наиболее важных признаков обе модели выделили три одинаковых показателя атмосфера, чистота и удобство. 
Для решения задачи регрессии лучше всего подходит метод Лассо-регрессии. 

Решили задачу кластеризации через методы t-SNE и MiniSom. 
Первый метод выявил только один результирующий признак дистанцию 
от центра города, второй метод выявил три результирующих признака дистанция, удобство и чистота. 

Решили задачу классификации через методы RandomForestClassifier и GradientBoostingClassifier. 
Точность первого метода - 95.65%, точность второго метода - 92.57%. 
Таким образом, RandomForestClassifier может считаться более эффективным методом

---

