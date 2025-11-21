
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
    #return(response.text)
    
    formatted_response = json.loads(response.text)
    print(formatted_response)
    #return(formatted_response)
    
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    for emotion, score in emotions.items():
        print(f"{emotion}: {score}")
    
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
    
    print(f"Dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1] * 100:.2f}%)")  

    return emotions  
    