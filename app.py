from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle as p
from fastapi import FastAPI

app = FastAPI()

modelfile = 'models/diabetes_final.pickle'    

model = p.load(open(modelfile, 'rb'))

@app.get('/')
def main():
    return ('Predict diabetes API')
    
@app.post('/api/')
def makecalc():
	j_data = request.get_json()

	prediction = np.array2string(model.predict(j_data))
	
	return jsonify(prediction)
