from flask import Flask, jsonify, request
import predict_wisata  # Pastikan modul ini ada dan berfungsi

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "API is running!"

@app.route('/api/hello', methods=['GET'])
def hello():
    response = {'message': 'Hello, World!'}
    return jsonify(response)

@app.route('/api/predictWisata', methods=['POST'])
def predictWisata():
    City = request.form["City"]

    predict_result = predict_wisata.predict_rating(City)
    response = {
        'data': predict_result
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
