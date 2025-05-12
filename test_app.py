import unittest
import json
import torch
from unittest.mock import patch, MagicMock
from app import app, prepare_input_data, load_models

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        load_models()  # Load any available models for the tests

        self.sample_input = {
            "creditScore": "600",
            "age": "40",
            "tenure": "3",
            "balance": "60000",
            "numProducts": "2",
            "hasCard": "1",
            "isActive": "1",
            "salary": "50000",
            "geography": "France",
            "gender": "Male"
        }

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_models_route(self):
        response = self.app.get('/models')
        self.assertEqual(response.status_code, 200)

    def test_prepare_input_data(self):
        input_array = prepare_input_data(self.sample_input)
        self.assertEqual(input_array.shape, (1, 13))
        self.assertIsInstance(input_array, object)

    @patch('app.xgb_model')
    @patch('app.tabnet_model')
    @patch('app.predict_vqc')
    @patch('app.predict_qnn')
    def test_predict_route(self, mock_qnn, mock_vqc, mock_tabnet_model, mock_xgb_model):
        # Mock all models to return 1
        mock_xgb_model.predict.return_value = [1]

        # Mock TabNet model
        mock_tabnet_model.eval = MagicMock()
        mock_tabnet_model.__call__ = MagicMock(return_value=torch.tensor([[0.7]]))

        # Mock VQC and QNN models
        mock_vqc.return_value = 1
        mock_qnn.return_value = 1

        response = self.app.post('/predict', data=json.dumps(self.sample_input), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('predictions', data)
        self.assertEqual(data['predictions']['consensus'], 1)

    def test_predict_missing_input(self):
        bad_input = {"creditScore": "600"}  # Missing many fields
        response = self.app.post('/predict', data=json.dumps(bad_input), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()

