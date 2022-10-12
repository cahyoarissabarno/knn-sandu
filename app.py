from flask import Flask, request, jsonify, make_response
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Membuat server flask
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
	return "<h1>Sandu KNN API</h1>"


########### GET GIZI STATUS ENPOINT ###########
@app.route('/get_gizi_status', methods=['POST'])
def get_gizi_status():
	try:
		data = request.json
		dataset1 = pd.read_csv("BB_U.csv")
		train_data_gz = np.array(dataset1)[:,0:-1]
		train_label_gz = np.array(dataset1)[:,-1]

		knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
		knn.fit(train_data_gz, train_label_gz)

		# print("Data train: ", train_data_gz)
		# print("Label train: ", train_label_gz)

		gender= data["gender"]
		umur = data["umur"]
		bb = data["bb"]

		test_data_gz= np.array([float(gender), float(umur), float(bb)])
		test_data_gz2 = np.reshape(test_data_gz,(1,-1))
		hasil = knn.predict(test_data_gz2)
		
	except Exception as e:
		print("Error: " + str(e))

	# return jsonify(hasil)
	return make_response(jsonify({"status":hasil[0]}),200)


########### GET BERAT STATUS ENPOINT ###########
@app.route('/get_berat_status', methods=['POST'])
def get_berat_status():
	try:
		data = request.json
		dataset3 = pd.read_csv("PB_BB.csv")
		train_data_BB = np.array(dataset3)[:,0:-1]
		train_label_BB = np.array(dataset3)[:,-1]

		knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
		knn.fit(train_data_BB, train_label_BB)

		gender= data["gender"]
		umur = data["umur"]
		tb = data["tb"]
		bb = data["bb"]

		test_data_TB = np.array([float(gender), float(umur), float(tb)])
		test_data_BB = np.array([float(gender), float(tb), float(bb)])
		test_data_BB = np.reshape(test_data_TB,(1,-1))
		hasil = knn.predict(test_data_BB)
		
	except Exception as e:
		print("Error: " + str(e))

	# return jsonify(hasil)
	return make_response(jsonify({"status":hasil[0]}),200)


########### GET TINGGI STATUS ENPOINT ###########
@app.route('/get_tinggi_status', methods=['POST'])
def get_tinggi_status():
	try:
		data = request.json
		dataset2 = pd.read_csv("PB_U.csv")
		train_data_TB = np.array(dataset2)[:,0:-1]
		train_label_TB = np.array(dataset2)[:,-1]

		knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
		knn.fit(train_data_TB, train_label_TB)

		gender= data["gender"]
		umur = data["umur"]
		tb = data["tb"]

		test_data_TB = np.array([float(gender), float(umur), float(tb)])
		test_data_TB = np.reshape(test_data_TB,(1,-1))
		hasil = knn.predict(test_data_TB)
		
	except Exception as e:
		print("Error: " + str(e))

	# return jsonify(hasil)
	return make_response(jsonify({"status":hasil[0]}),200)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5010, debug=False)
