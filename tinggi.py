from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

dataset2 = pd.read_csv("PB_U.csv")
train_data_TB = np.array(dataset2)[:,0:-1]
train_label_TB = np.array(dataset2)[:,-1]

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(train_data_TB, train_label_TB)

# print("Data train: ", train_data_TB)
# print("Label train: ", train_label_TB)

gender= input("Masukkan gender: ")
umur = input("Umur anak (bulan): ")
tb = input("Masukkan tinggi badan anak: ")

test_data_TB = np.array([float(gender), float(umur), float(tb)])
print(test_data_TB)

test_data_TB = np.reshape(test_data_TB,(1,-1))
print(test_data_TB)
hasil = knn.predict(test_data_TB)
print("Hasil dari k-NN: ", hasil)