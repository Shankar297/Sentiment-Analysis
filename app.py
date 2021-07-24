from flask import Flask, render_template, request, url_for
import joblib

vec = joblib.load('tfidf_model.pkl')
model = joblib.load('new_model.pkl')

app = Flask('__name__')

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/check',methods = ['POST'])
def check():
    if request.method == 'POST':
        received_text = str(request.form['rawtext'])
        new_text = []
        new_text.append(received_text)
        vector = vec.transform(new_text)
        result= model.predict(vector)[0]
        if result==0:
            output = 'Negative Sentiment'
        else:
            output = 'Positive Sentiment'

    return render_template('home.html', your_text = received_text, result = output)
    

if __name__ =='__main__':
    app.run(debug=True)
