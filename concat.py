import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data-1711189329586.csv")
# print(data)

df1 = data[(data["target"]==1) & data["mip"].between(10,100)]
# print(df1)

df2 = data[(data["target"]==0) & data["mip"].between(100,105)]
# print(df2)

filtered_data = pd.concat([df1,df2])
# print(df_combined)

print("Mean: ",filtered_data.mean())
print(filtered_data.max())
filtered_data.max(), filtered_data.min()
filtered_data = filtered_data.sort_values(by="sip", ascending = True)
#Exersize1
X_train, X_test, y_train, y_test = train_test_split(filtered_data.drop('target', axis=1),filtered_data['target'],test_size=0.2 ,random_state=33, stratify=filtered_data['target'], )
#
# Exersize2
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
data_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)

print(f"Максимальное значение STDC из тренировочной выборки: {X_train['stdc'].max():.3f}")
print(f"Выборочное среднее для столбца STDIP из тренировочной выборки (после нормировки): {data_scaled['stdip'].mean():.3f}")

# Exersize4(логистическая регрессия)
clf = LogisticRegression().fit(X_train_scaled, y_train)
pred = clf.predict(X_test_scaled)
print("первая матрица: ", confusion_matrix(pred, y_test))

# Exersize5 - метод ближайших соседей
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
pred = knn.predict(X_test_scaled)
print("вторая матрица: ", confusion_matrix(pred, y_test))
print(f1_score(pred, y_test))
