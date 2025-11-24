import requests
import json

def emotion_detector(text_to_analyse):
    # URL of the emotion_detection service
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    # Constructing the request payload in the expected format
    myobj = { "raw_document": { "text": text_to_analyse } }
    
    # Custom header specifying the model ID for the emotion_detection service
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    
    # Sending a POST request to the emotion_detection API
    response = requests.post(url, json=myobj, headers=header)

    # format the response
    formatted_response = json.loads(response.text)

    #Extract emotions
    emotions = formatted_response['emotionPredictions'][0]['emotion']

    #Find the dominant emotion
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]

    #Create result dictionary with all emotions plus dominant emotion
    result = {**emotions, "dominant_emotion":dominant_emotion}

    return result