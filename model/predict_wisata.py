import numpy as np  # untuk operasi numerik dan komputasi array multidimensi secara efisien.
import pandas as pd  # untuk manipulasi dan analisis data 
import pickle  # untuk serialisasi dan deserialisasi objek Python.
import tensorflow as tf  # untuk machine learning dan neural network.

# Memuat model yang telah dilatih
loaded_model = tf.keras.models.load_model('data_wisata.h5')

# Memuat skalar yang telah disimpan
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Memuat data model
data_model = pd.read_csv('tourism_with_id.csv')

def predict_rating(City):
    # Memfilter data berdasarkan kota
    filtered_data = data_model[data_model['City'] == City]
    
    # Jika tidak ada data untuk kota yang diberikan, kembalikan pesan kosong
    if filtered_data.empty:
        return "Tidak ada data untuk kota tersebut."

    # Mengambil lokasi (Long dan Lat) dari data yang difilter
    locations = filtered_data[['Long', 'Lat']].values

    # Melakukan penskalaan terhadap lokasi menggunakan skalar yang telah dimuat sebelumnya
    scaled_locations = loaded_scaler.transform(locations)

    # Memprediksi rating menggunakan model yang telah dimuat sebelumnya 
    predicted_ratings = loaded_model.predict(scaled_locations)

    # Mengonversi hasil prediksi rating dari format kategorikal ke rating asli dengan mengambil argumen terbesar dari hasil prediksi dan menambahkan 1.
    predicted_ratings = np.argmax(predicted_ratings, axis=-1) + 1

    # Menambahkan hasil prediksi ke data yang difilter
    filtered_data['predictedRating'] = predicted_ratings

    # Mengambil 5 tempat dengan rating tertinggi
    top_rated_places = filtered_data.nlargest(5, 'Rating')

    # Menyusun hasil prediksi ke dalam format list of dictionaries
    results = []
    for _, row in top_rated_places.iterrows():
        result = {
            'City': row['City'],
            # 'displayName': row['displayName'],
            'Place_Name': row['Place_Name'],
            'Long': row['Long'],
            'Lat': row['Lat'],
            'Rating': row['Rating'],
            'predictedRating': row['predictedRating']
        }
        results.append(result)

    return results
