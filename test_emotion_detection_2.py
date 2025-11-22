import unittest
from unittest.mock import patch, Mock, MagicMock
import json
import requests


# ============================================
# ORIGINAL CODE (emotion_detector.py)
# ============================================

def emotion_detector(text_to_analyse):
    """
    Detect emotions in text using Watson NLP API
    
    Args:
        text_to_analyse: String of text to analyze
        
    Returns:
        Dictionary of emotion scores
    """
    # URL of the emotion_detection service
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    # Constructing the request payload in the expected format
    myobj = {"raw_document": {"text": text_to_analyse}}
    
    # Custom header specifying the model ID for the emotion_detection service
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    
    # Sending a POST request to the emotion_detection API
    response = requests.post(url, json=myobj, headers=header)
    
    formatted_response = json.loads(response.text)
    print(formatted_response)
    
    emotions = formatted_response['emotionPredictions'][0]['emotion']
    for emotion, score in emotions.items():
        print(f"{emotion}: {score}")
    
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
    
    print(f"Dominant emotion: {dominant_emotion[0]} ({dominant_emotion[1] * 100:.2f}%)")  

    return emotions


# ============================================
# UNIT TESTS
# ============================================

class TestEmotionDetector(unittest.TestCase):
    """Comprehensive test suite for emotion_detector function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.0132405795,
                    'disgust': 0.0020517302,
                    'fear': 0.009090992,
                    'joy': 0.9699522,
                    'sadness': 0.054984167
                },
                'target': '',
                'emotionMentions': [{
                    'span': {
                        'begin': 0,
                        'end': 21,
                        'text': 'I love new technology'
                    },
                    'emotion': {
                        'anger': 0.0132405795,
                        'disgust': 0.0020517302,
                        'fear': 0.009090992,
                        'joy': 0.9699522,
                        'sadness': 0.054984167
                    }
                }]
            }],
            'producerId': {
                'name': 'Ensemble Aggregated Emotion Workflow',
                'version': '0.0.1'
            }
        }
        
        self.expected_url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
        self.expected_headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    
    # ========================================
    # Tests for Successful API Calls
    # ========================================
    
    @patch('requests.post')
    def test_emotion_detector_returns_emotions(self, mock_post):
        """Test that function returns emotion dictionary"""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        # Call function
        result = emotion_detector("I love new technology")
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 5)
    
    @patch('requests.post')
    def test_emotion_detector_contains_all_emotions(self, mock_post):
        """Test that result contains all 5 emotion types"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I love new technology")
        
        expected_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness']
        for emotion in expected_emotions:
            self.assertIn(emotion, result)
    
    @patch('requests.post')
    def test_emotion_detector_correct_api_url(self, mock_post):
        """Test that correct API URL is called"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        emotion_detector("test text")
        
        # Verify the URL was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], self.expected_url)
    
    @patch('requests.post')
    def test_emotion_detector_correct_headers(self, mock_post):
        """Test that correct headers are sent"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        emotion_detector("test text")
        
        # Verify headers
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs['headers'], self.expected_headers)
    
    @patch('requests.post')
    def test_emotion_detector_correct_payload(self, mock_post):
        """Test that correct payload is sent"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        test_text = "I am very happy today"
        emotion_detector(test_text)
        
        # Verify payload
        call_kwargs = mock_post.call_args[1]
        expected_payload = {"raw_document": {"text": test_text}}
        self.assertEqual(call_kwargs['json'], expected_payload)
    
    @patch('requests.post')
    def test_emotion_detector_joy_dominant(self, mock_post):
        """Test detection of joy as dominant emotion"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I love new technology")
        
        # Joy should be the highest score
        max_emotion = max(result.items(), key=lambda x: x[1])
        self.assertEqual(max_emotion[0], 'joy')
        self.assertGreater(result['joy'], 0.9)
    
    @patch('requests.post')
    def test_emotion_detector_with_different_texts(self, mock_post):
        """Test function with various input texts"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        test_cases = [
            "I am glad this happened",
            "I am really mad about this",
            "I feel disgusted just hearing about this",
            "I am so sad about this",
            "I am really afraid that this will happen",
            ""
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                result = emotion_detector(text)
                self.assertIsInstance(result, dict)
    
    @patch('requests.post')
    def test_emotion_scores_are_floats(self, mock_post):
        """Test that all emotion scores are float type"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("test text")
        
        for emotion, score in result.items():
            self.assertIsInstance(score, float)
    
    @patch('requests.post')
    def test_emotion_scores_in_valid_range(self, mock_post):
        """Test that scores are between 0 and 1"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("test text")
        
        for emotion, score in result.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    # ========================================
    # Tests for Specific Statements with Expected Dominant Emotions
    # ========================================
    
    @patch('requests.post')
    def test_statement_glad_this_happened_joy(self, mock_post):
        """Test: 'I am glad this happened' -> joy"""
        joy_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.01,
                    'disgust': 0.005,
                    'fear': 0.008,
                    'joy': 0.92,
                    'sadness': 0.057
                }
            }]

        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(joy_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I am glad this happened")
        
        # Verify joy is dominant
        dominant = max(result.items(), key=lambda x: x[1])
        self.assertEqual(dominant[0], 'joy', 
                        f"Expected 'joy' but got '{dominant[0]}' for 'I am glad this happened'")
        self.assertGreater(result['joy'], 0.5, "Joy score should be > 0.5")
    
    @patch('requests.post')
    def test_statement_really_mad_anger(self, mock_post):
        """Test: 'I am really mad about this' -> anger"""
        anger_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.88,
                    'disgust': 0.03,
                    'fear': 0.02,
                    'joy': 0.01,
                    'sadness': 0.06
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(anger_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I am really mad about this")
        
        # Verify anger is dominant
        dominant = max(result.items(), key=lambda x: x[1])
        self.assertEqual(dominant[0], 'anger',
                        f"Expected 'anger' but got '{dominant[0]}' for 'I am really mad about this'")
        self.assertGreater(result['anger'], 0.5, "Anger score should be > 0.5")
    
    @patch('requests.post')
    def test_statement_feel_disgusted_disgust(self, mock_post):
        """Test: 'I feel disgusted just hearing about this' -> disgust"""
        disgust_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.15,
                    'disgust': 0.75,
                    'fear': 0.03,
                    'joy': 0.01,
                    'sadness': 0.06
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(disgust_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I feel disgusted just hearing about this")
        
        # Verify disgust is dominant
        dominant = max(result.items(), key=lambda x: x[1])
        self.assertEqual(dominant[0], 'disgust',
                        f"Expected 'disgust' but got '{dominant[0]}' for 'I feel disgusted just hearing about this'")
        self.assertGreater(result['disgust'], 0.5, "Disgust score should be > 0.5")
    
    @patch('requests.post')
    def test_statement_so_sad_sadness(self, mock_post):
        """Test: 'I am so sad about this' -> sadness"""
        sadness_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.02,
                    'disgust': 0.01,
                    'fear': 0.04,
                    'joy': 0.01,
                    'sadness': 0.92
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(sadness_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I am so sad about this")
        
        # Verify sadness is dominant
        dominant = max(result.items(), key=lambda x: x[1])
        self.assertEqual(dominant[0], 'sadness',
                        f"Expected 'sadness' but got '{dominant[0]}' for 'I am so sad about this'")
        self.assertGreater(result['sadness'], 0.5, "Sadness score should be > 0.5")
    
    @patch('requests.post')
    def test_statement_really_afraid_fear(self, mock_post):
        """Test: 'I am really afraid that this will happen' -> fear"""
        fear_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.02,
                    'disgust': 0.01,
                    'fear': 0.86,
                    'joy': 0.01,
                    'sadness': 0.10
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(fear_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("I am really afraid that this will happen")
        
        # Verify fear is dominant
        dominant = max(result.items(), key=lambda x: x[1])
        self.assertEqual(dominant[0], 'fear',
                        f"Expected 'fear' but got '{dominant[0]}' for 'I am really afraid that this will happen'")
        self.assertGreater(result['fear'], 0.5, "Fear score should be > 0.5")
    
    # ========================================
    # Test All Statements Together
    # ========================================
    
    @patch('requests.post')
    def test_all_required_statements(self, mock_post):
        """Test all 5 required statements with expected dominant emotions"""
        test_cases = [
            {
                'statement': 'I am glad this happened',
                'expected_emotion': 'joy',
                'response': {
                    'emotionPredictions': [{
                        'emotion': {
                            'anger': 0.01,
                            'disgust': 0.005,
                            'fear': 0.008,
                            'joy': 0.92,
                            'sadness': 0.057
                        }
                    }]
                }
            },
            {
                'statement': 'I am really mad about this',
                'expected_emotion': 'anger',
                'response': {
                    'emotionPredictions': [{
                        'emotion': {
                            'anger': 0.88,
                            'disgust': 0.03,
                            'fear': 0.02,
                            'joy': 0.01,
                            'sadness': 0.06
                        }
                    }]
                }
            },
            {
                'statement': 'I feel disgusted just hearing about this',
                'expected_emotion': 'disgust',
                'response': {
                    'emotionPredictions': [{
                        'emotion': {
                            'anger': 0.15,
                            'disgust': 0.75,
                            'fear': 0.03,
                            'joy': 0.01,
                            'sadness': 0.06
                        }
                    }]
                }
            },
            {
                'statement': 'I am so sad about this',
                'expected_emotion': 'sadness',
                'response': {
                    'emotionPredictions': [{
                        'emotion': {
                            'anger': 0.02,
                            'disgust': 0.01,
                            'fear': 0.04,
                            'joy': 0.01,
                            'sadness': 0.92
                        }
                    }]
                }
            },
            {
                'statement': 'I am really afraid that this will happen',
                'expected_emotion': 'fear',
                'response': {
                    'emotionPredictions': [{
                        'emotion': {
                            'anger': 0.02,
                            'disgust': 0.01,
                            'fear': 0.86,
                            'joy': 0.01,
                            'sadness': 0.10
                        }
                    }]
                }
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(statement=test_case['statement']):
                # Setup mock
                mock_response = Mock()
                mock_response.text = json.dumps(test_case['response'])
                mock_post.return_value = mock_response
                
                # Call function
                result = emotion_detector(test_case['statement'])
                
                # Verify dominant emotion
                dominant = max(result.items(), key=lambda x: x[1])
                self.assertEqual(
                    dominant[0],
                    test_case['expected_emotion'],
                    f"Statement: '{test_case['statement']}' - "
                    f"Expected '{test_case['expected_emotion']}' but got '{dominant[0]}'"
                )
    
    # ========================================
    # Tests for Different Emotion Scenarios
    # ========================================
    
    @patch('requests.post')
    def test_mixed_emotions(self, mock_post):
        """Test with mixed emotion scores"""
        mixed_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.2,
                    'disgust': 0.2,
                    'fear': 0.2,
                    'joy': 0.2,
                    'sadness': 0.2
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(mixed_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("Mixed feelings")
        
        # All emotions should be equal
        scores = list(result.values())
        self.assertTrue(all(s == scores[0] for s in scores))
    
    # ========================================
    # Tests for Error Handling
    # ========================================
    
    @patch('requests.post')
    def test_api_connection_error(self, mock_post):
        """Test handling of connection errors"""
        mock_post.side_effect = requests.ConnectionError("Connection failed")
        
        with self.assertRaises(requests.ConnectionError):
            emotion_detector("test text")
    
    @patch('requests.post')
    def test_api_timeout_error(self, mock_post):
        """Test handling of timeout errors"""
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        with self.assertRaises(requests.Timeout):
            emotion_detector("test text")
    
    @patch('requests.post')
    def test_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON response"""
        mock_response = Mock()
        mock_response.text = "Invalid JSON"
        mock_post.return_value = mock_response
        
        with self.assertRaises(json.JSONDecodeError):
            emotion_detector("test text")
    
    @patch('requests.post')
    def test_missing_emotion_predictions_key(self, mock_post):
        """Test handling of missing emotionPredictions key"""
        invalid_response = {'producerId': {'name': 'test'}}
        
        mock_response = Mock()
        mock_response.text = json.dumps(invalid_response)
        mock_post.return_value = mock_response
        
        with self.assertRaises(KeyError):
            emotion_detector("test text")
    
    @patch('requests.post')
    def test_empty_emotion_predictions(self, mock_post):
        """Test handling of empty emotionPredictions array"""
        empty_response = {'emotionPredictions': []}
        
        mock_response = Mock()
        mock_response.text = json.dumps(empty_response)
        mock_post.return_value = mock_response
        
        with self.assertRaises(IndexError):
            emotion_detector("test text")
    
    @patch('requests.post')
    def test_api_500_error(self, mock_post):
        """Test handling of server errors"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = json.dumps({'error': 'Internal Server Error'})
        mock_post.return_value = mock_response
        
        # Function doesn't check status code, but we can test the behavior
        with self.assertRaises((KeyError, json.JSONDecodeError)):
            emotion_detector("test text")
    
    # ========================================
    # Tests for Edge Cases
    # ========================================
    
    @patch('requests.post')
    def test_empty_string_input(self, mock_post):
        """Test with empty string input"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        result = emotion_detector("")
        
        # Should still work, API handles empty text
        self.assertIsInstance(result, dict)
        
        # Verify empty string was sent
        call_kwargs = mock_post.call_args[1]
        self.assertEqual(call_kwargs['json']['raw_document']['text'], "")
    
    @patch('requests.post')
    def test_very_long_text(self, mock_post):
        """Test with very long text input"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        long_text = "This is a test. " * 1000  # 1000 repetitions
        result = emotion_detector(long_text)
        
        self.assertIsInstance(result, dict)
    
    @patch('requests.post')
    def test_special_characters_in_text(self, mock_post):
        """Test with special characters"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        special_text = "Hello! @#$%^&*() 你好 émotions"
        result = emotion_detector(special_text)
        
        self.assertIsInstance(result, dict)
    
    @patch('requests.post')
    def test_unicode_text(self, mock_post):
        """Test with unicode characters"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        unicode_text = "I ❤️ Python 😊"
        result = emotion_detector(unicode_text)
        
        self.assertIsInstance(result, dict)
    
    # ========================================
    # Tests for Print Output (Optional)
    # ========================================
    
    @patch('requests.post')
    @patch('builtins.print')
    def test_print_statements_called(self, mock_print, mock_post):
        """Test that print statements are executed"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        emotion_detector("test text")
        
        # Verify print was called multiple times
        self.assertGreater(mock_print.call_count, 0)
    
    @patch('requests.post')
    @patch('builtins.print')
    def test_dominant_emotion_printed(self, mock_print, mock_post):
        """Test that dominant emotion is printed correctly"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        emotion_detector("test text")
        
        # Check that dominant emotion print was called
        print_calls = [str(call) for call in mock_print.call_args_list]
        dominant_printed = any('Dominant emotion' in str(call) for call in print_calls)
        self.assertTrue(dominant_printed)
    
    # ========================================
    # Integration-style Tests
    # ========================================
    
    @patch('requests.post')
    def test_full_workflow(self, mock_post):
        """Test complete workflow from input to output"""
        mock_response = Mock()
        mock_response.text = json.dumps(self.sample_response)
        mock_post.return_value = mock_response
        
        input_text = "I love new technology"
        
        # Call function
        result = emotion_detector(input_text)
        
        # Verify complete workflow
        # 1. API was called
        mock_post.assert_called_once()
        
        # 2. Correct URL and headers
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], self.expected_url)
        self.assertEqual(call_args[1]['headers'], self.expected_headers)
        
        # 3. Correct payload
        expected_payload = {"raw_document": {"text": input_text}}
        self.assertEqual(call_args[1]['json'], expected_payload)
        
        # 4. Result is valid
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 5)
        
        # 5. Joy is dominant
        max_emotion = max(result.items(), key=lambda x: x[1])
        self.assertEqual(max_emotion[0], 'joy')


# ============================================
# PERFORMANCE TESTS (Optional)
# ============================================

class TestEmotionDetectorPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    @patch('requests.post')
    def test_multiple_consecutive_calls(self, mock_post):
        """Test multiple consecutive API calls"""
        sample_response = {
            'emotionPredictions': [{
                'emotion': {
                    'anger': 0.1,
                    'disgust': 0.1,
                    'fear': 0.1,
                    'joy': 0.5,
                    'sadness': 0.2
                }
            }]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(sample_response)
        mock_post.return_value = mock_response
        
        # Make 100 consecutive calls
        for i in range(100):
            result = emotion_detector(f"Test text {i}")
            self.assertIsInstance(result, dict)
        
        self.assertEqual(mock_post.call_count, 100)


# ============================================
# RUN TESTS
# ============================================

if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)