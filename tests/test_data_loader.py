import unittest
from data.loaders.ucf50_loader import load_ucf50_data

class TestDataLoader(unittest.TestCase):
    def test_load_ucf50_data(self):
        data_dir = '/path/to/ucf50/data'
        X, y = load_ucf50_data(data_dir, sample_per_vid_passed=10)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertGreater(X.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
