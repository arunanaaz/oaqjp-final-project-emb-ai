from EmotionDetection.emotion_detection import emotion_detector
import unittest
import json

class TestEmotionDetector(unittest.TestCase):
    def test_emotion_detector(self):
        
        # Test case for joy emotion
        result_1 = emotion_detector('I am glad this happened')

          # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 5)


        #formatted_response = json.loads(response.text)
        #print(formatted_response)

        #result = extract_emotions(formatted_response)
        #self.assertIsNotNone(result)
        #self.assertIn('anger', result)
        #self.assertIn('disgust', result)
        #self.assertIn('fear', result)
        #self.assertIn('joy', result)
        #self.assertIn('sadness', result)



        #emotions = formatted_response['emotionPredictions'][0]['emotion']
        #for emotion, score in emotions.items():
            #print(f"{emotion}: {score}")
        #self.assertEqual(result_1['emotionPredictions'][0]['emotion'], 'joy')
        
        # Test case for anger emotion
        #result_2 = emotion_detector('I am really mad about this')
        #self.assertEqual(result_2['emotionPredictions'][0]['anger'], 'anger')
        
        # Test case for disgust emotion
        #result_3 = emotion_detector('I feel disgusted just hearing about this')
        #self.assertEqual(result_3['emotionPredictions'][0]['disgust'], 'disgust')


    def extract_emotions(data):
        try:
            # Handle JSON string input
            if isinstance(data, str):
                data = json.loads(data)
            
            # Navigate to emotion scores
            predictions = data.get('emotionPredictions', [])
            if not predictions:
                return None
                
            emotions = predictions[0].get('emotion', {})
            return emotions if emotions else None
            
        except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError):
            return None    


    def get_text_from_span(data):
        try:
            if isinstance(data, str):
                data = json.loads(data)
                
            mentions = get_emotion_mentions(data)
            if mentions and len(mentions) > 0:
                return mentions[0].get('span', {}).get('text')
            return None
            
        except (KeyError, IndexError, TypeError, AttributeError):
            return None    

    unittest.main()