"""
Flask application for emotion detection analysis.
"""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Analyzer")

@app.route("/emotionDetector")
def emotion_analyzer():
    """
    Analyze emotion from text provided in request arguments.

    Returns:
        JSON response with emotion analysis results or error message
    """
    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')

    # Handle empty or missing input
    if not text_to_analyze or text_to_analyze.strip() == '':
        return "Invalid text! Please try again!", 400

    # Pass the text to the emotion_detector function and store the response
    response = emotion_detector(text_to_analyze)

    # Check if the response contains None for dominant_emotion (error case)
    if response.get('dominant_emotion') is None:
        return "Invalid text! Please try again!", 400

    # Format the response message
    anger = response['anger']
    disgust = response['disgust']
    fear = response['fear']
    joy = response['joy']
    sadness = response['sadness']
    dominant_emotion = response['dominant_emotion']

    return (f"For the given statement, the system response is 'anger': {anger}, "
            f"'disgust': {disgust}, 'fear': {fear}, 'joy': {joy}, "
            f"'sadness': {sadness}. The dominant emotion is {dominant_emotion}.")

@app.route("/")
def render_index_page():
    """
    Render the main index page.

    Returns:
        Rendered HTML template for index page
    """
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    