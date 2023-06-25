from flask import Flask,render_template,request
import numpy as np
import pickle

model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_house_price():
    floors = float(request.form.get('floors'))
    waterfront = int(request.form.get('waterfront'))
    bedrooms = int(request.form.get('bedrooms'))
    sqft_basement = int(request.form.get('sqft_basement'))
    view = int(request.form.get('view'))
    bathrooms = float(request.form.get('bathrooms'))
    sqft_above = int(request.form.get('sqft_above'))
    sqft_living = int(request.form.get('sqft_living'))

    #prediction
    result=model.predict(np.array([floors,waterfront,bedrooms,sqft_basement,view,bathrooms,sqft_above,sqft_living]).reshape(1,8))
    
    return render_template('index.html',result=result)

if __name__=='__main__':
    app.run(debug=True)


