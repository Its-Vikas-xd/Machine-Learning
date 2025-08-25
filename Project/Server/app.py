from flask import Flask, request, jsonify
import util   # make sure util.py is in same folder or in PYTHONPATH

app = Flask(__name__)


@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')  # allows frontend requests
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server for Price Prediction....")
    app.run(debug=True)
