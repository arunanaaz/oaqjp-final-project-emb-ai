import requests
import json


def emotion_detector(text_to_analyse):
    # Check for empty or None input
    if not text_to_analyse or text_to_analyse.strip() == "":
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
    
    try:
        # URL of the emotion_detection service
        url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
        
        # Constructing the request payload in the expected format
        myobj = { "raw_document": { "text": text_to_analyse } }
        
        # Custom header specifying the model ID for the emotion_detection service
        header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
        
        # Sending a POST request to the emotion_detection API
        response = requests.post(url, json=myobj, headers=header, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response
            formatted_response = json.loads(response.text)
            
            # Extract emotions
            emotions = formatted_response['emotionPredictions'][0]['emotion']
            
            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Create result dictionary with all emotions plus dominant_emotion
            result = {**emotions, "dominant_emotion": dominant_emotion}
            
            return result
        
        elif response.status_code == 400:
            # Bad request - invalid input
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None
            }
        
        elif response.status_code == 500:
            # Server error
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None
            }
        
        else:
            # Other status codes
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None
            }
    
    except requests.exceptions.Timeout:
        # Handle timeout error
        print("Error: Request timed out")
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
    
    except requests.exceptions.ConnectionError:
        # Handle connection error
        print("Error: Unable to connect to the service")
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
    
    except (KeyError, IndexError, ValueError) as e:
        # Handle errors in parsing the response
        print(f"Error: Unable to parse response - {str(e)}")
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }
    
    except Exception as e:
        # Handle any other unexpected errors
        print(f"Unexpected error: {str(e)}")
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }