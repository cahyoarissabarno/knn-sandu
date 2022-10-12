from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

dataset3 = pd.read_csv("PB_BB.csv")
train_data_BB = np.array(dataset3)[:,0:-1]
train_label_BB = np.array(dataset3)[:,-1]

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(train_data_BB, train_label_BB)

# print("Data train: ", train_data_BB)
# print("Label train: ", train_label_BB)

gender= input("Masukkan gender: ")
umur = input("Umur anak (bulan): ")
tb = input("Masukkan tinggi badan anak: ")
bb = input("Masukkan berat badan: ")

test_data_TB = np.array([float(gender), float(umur), float(tb)])
print(test_data_TB)

test_data_BB = np.array([float(gender), float(tb), float(bb)])
print(test_data_BB)

test_data_BB = np.reshape(test_data_TB,(1,-1))
print(test_data_BB)
hasil = knn.predict(test_data_BB)
print("Hasil dari k-NN: ", hasil)