import pandas as pd
import numpy as np
import pickle
import re
import flask
from flask import request, jsonify
#from functions import data_prep_predict

# Create app
app = flask.Flask(__name__)

# Define path
ADD_DATA_PATH = r"C:\Users\IS\Documents\Data Scientist\P7\P7_semionov_irina\P7_02_dossier\\"

# Define external data
cols_2_keep = pickle.load(open(ADD_DATA_PATH + 'src\cols_2_keep.sav', 'rb'))
clf = pickle.load(open(ADD_DATA_PATH + 'src\scoring_model.sav', 'rb'))

def data_prep_predict(df, cols_2_keep):

    # Filling NaN values with 0
    df = df.fillna(0)

    # Define categorical feats
    categorical_feats = [f for f in df.columns if df[f].dtype == 'object']
    for f_ in categorical_feats:
        df[f_], indexer = pd.factorize(df[f_])

    # Checking the presence of each column (for prediction)
    for col in cols_2_keep:
        if col not in df.columns:
            df[col] = 0

    # Normalize headers
    df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    return df[cols_2_keep]


def compute_score(data):
    data = {key:[value] for key, value in data.items()}
    data = pd.DataFrame(data)
    data = data_prep_predict(data, cols_2_keep)
    ids_data = data['SK_ID_CURR']
    prediction = np.zeros(data.shape[0])
    feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
    prediction += clf.predict_proba(data[feats], num_iteration=clf.best_iteration_)[:, 1]
    score = prediction[0]
    customer_id = ids_data[0]
    score = float(score)
    customer_id = int(customer_id)
    
    return jsonify({
        "customer_id":customer_id,
        "score":score
    })

@app.route('/', methods=['GET'])
def home():
    return "Prediction API."


@app.route('/scoring', methods=['POST'])
def customer_score():
    data = request.get_json()
    print('data', data)
    #ids_data, score = compute_score(data)
    return compute_score(data)


if __name__ == "__main__":
    app.run(debug=True)