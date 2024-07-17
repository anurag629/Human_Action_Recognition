import unittest
from src.models.cnn_models import build_model_1, build_model_2, build_model_3

class TestModels(unittest.TestCase):
    def test_build_model_1(self):
        model = build_model_1((64, 64, 3), 50)
        self.assertEqual(model.output_shape[1], 50)
    
    def test_build_model_2(self):
        model = build_model_2((64, 64, 3), 50)
        self.assertEqual(model.output_shape[1], 50)
    
    def test_build_model_3(self):
        model = build_model_3((64, 64, 3), 50)
        self.assertEqual(model.output_shape[1], 50)

if __name__ == '__main__':
    unittest.main()
