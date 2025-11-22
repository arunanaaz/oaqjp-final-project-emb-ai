from flask import Flask, render_template, request, jsonify
from EmotionDetection.emotion_detection_new import emotion_detector

app = Flask("Emotion Analyzer")

@app.route("/emotionDetector")
def emotion_analyzer():
    # Retrieve the text to analyze from the request arguments
    text_to_analyze = request.args.get('textToAnalyze')
    
    # Validate input
    if not text_to_analyze:
        return jsonify({"error": "No text provided"}), 400
    
    # Pass the text to the emotion_detector function and store the response
    response = emotion_detector(text_to_analyze)
    
    # Return the response as JSON
    return jsonify(response)


@app.route("/")
def render_index_page():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)