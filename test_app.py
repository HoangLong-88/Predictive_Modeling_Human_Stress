import unittest
from unittest.mock import patch

import app as web


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()

    def test_initial_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Start with a few words.', response.data)

    def test_invalid_inputs(self):
        for text, message in [('', b'Please enter'), ('a' * 5001, b'at most 5000'),
                              ('123 !!!', b'include some words')]:
            response = self.client.post('/', data={'text': text})
            self.assertEqual(response.status_code, 200)
            self.assertIn(message, response.data)
            self.assertNotIn(b'role="meter"', response.data)

    def test_prediction_and_escaping(self):
        response = self.client.post('/', data={'text': '<script>alert(1)</script> I feel overwhelmed by work.'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'role="meter"', response.data)
        self.assertIn(b'&lt;script&gt;', response.data)
        self.assertNotIn(b'<script>alert(1)</script>', response.data)

    def test_unavailable_model(self):
        with patch.object(web, 'model', None):
            response = self.client.post('/', data={'text': 'A difficult day'})
        self.assertIn(b'temporarily unavailable', response.data)

    def test_probability_maps_to_stress_class(self):
        class ReversedClasses:
            classes_ = [1, 0]
            def predict(self, texts):
                return [1]
            def predict_proba(self, texts):
                return [[0.8, 0.2]]
        with patch.object(web, 'model', ReversedClasses()):
            response = self.client.post('/', data={'text': 'A difficult day'})
        self.assertIn(b'aria-valuenow="80.0"', response.data)


if __name__ == '__main__':
    unittest.main()
