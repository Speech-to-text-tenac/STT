import unittest
from preprocessing import resample
import pandas as pd


data = pd.read_csv("../data/test.csv")
col = [] # columns to be filled with data attributes

class TestResampling(unittest.TestCase):
    def test_check_resampling(self):
        sampled_audio, rates = resample(data, 'key')
        expected = [44100, 44100, 44100, 44100, 44100]
        self.assertEqual(rates[8:13], expected)

if __name__=="__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
