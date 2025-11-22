import unittest
from server import app
import json


class TestEmotionDetector(unittest.TestCase):
    
    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        self.app.testing = True
    
    def test_joy_emotion(self):
        # Test with "I love my life"
        response = self.app.get('/emotionDetector?textToAnalyze=I love my life')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('joy', data)
        self.assertIn('dominant_emotion', data)
        self.assertEqual(data['dominant_emotion'], 'joy')
    
    def test_anger_emotion(self):
        # Test with "I don't like this"
        response = self.app.get('/emotionDetector?textToAnalyze=I don\'t like this')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('anger', data)
        self.assertIn('dominant_emotion', data)
    
    def test_disgust_emotion(self):
        # Test with "This is disgusting"
        response = self.app.get('/emotionDetector?textToAnalyze=This is disgusting')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('disgust', data)
        self.assertIn('dominant_emotion', data)
    
    def test_sadness_emotion(self):
        # Test with "I am so sad about this"
        response = self.app.get('/emotionDetector?textToAnalyze=I am so sad about this')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('sadness', data)
        self.assertIn('dominant_emotion', data)
    
    def test_fear_emotion(self):
        # Test with "I am really afraid"
        response = self.app.get('/emotionDetector?textToAnalyze=I am really afraid')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('fear', data)
        self.assertIn('dominant_emotion', data)
    
    def test_empty_input(self):
        # Test with empty input
        response = self.app.get('/emotionDetector?textToAnalyze=')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    def test_no_input(self):
        # Test with no textToAnalyze parameter
        response = self.app.get('/emotionDetector')
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()