from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import json

sia = SentimentIntensityAnalyzer()

app = Flask(__name__)


@app.route('/')
def new():
	result = ""
	return render_template('index.html', result=result)


@app.route('/sentiment', methods=['POST', 'GET'])
def sentiment():
	if request.method == 'POST': 
		sentence = request.form['Name']
		intensity = sia.polarity_scores(sentence)
		polarity_subjectivity = TextBlob(sentence).sentiment._asdict()
		intensity.update(polarity_subjectivity)

		with open('result.json', 'w') as f:
			json.dump(intensity, f)

		return render_template('index.html', result=intensity)
	else:
		return render_template('index.html')


if __name__ == '__main__':
	app.debug = True
	app.run()
