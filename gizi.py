from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

dataset1 = pd.read_csv("BB_U.csv")
train_data_gz = np.array(dataset1)[:,0:-1]
train_label_gz = np.array(dataset1)[:,-1]

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(train_data_gz, train_label_gz)

# print("Data train: ", train_data_gz)
# print("Label train: ", train_label_gz)

gender= input("Masukkan gender: ")
umur = input("Umur anak (bulan): ")
bb = input("Masukkan berat badan anak: ")

test_data_gz= np.array([float(gender), float(umur), float(bb)])
print(test_data_gz)

test_data_gz2 = np.reshape(test_data_gz,(1,-1))
hasil = knn.predict(test_data_gz2)
print("Hasil dari k-NN: ", hasil)