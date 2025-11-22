from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

import json


app = Flask("Emotion Analyzer")

@app.route("/emotionDetector")
def emotion_analyzer():
    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')
    # Pass the text to the sentiment_analyzer function and store the response
    response = emotion_detector(text_to_analyze)

    formatted_response = json.loads(response.text)
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    for emotion, score in emotions.items():
        print(f"{emotion}: {score}")


@app.route("/")
def render_index_page():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




   
    

    # Return a formatted string with the sentiment label and score
    #return "The given text has been identified as {} with a score of {}.".format(label.split('_')[1], score)
