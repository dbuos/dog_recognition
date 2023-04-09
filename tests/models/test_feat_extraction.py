import unittest
import torch
from drecg.models.feat_extraction import DiffFeatureDetectorParam


class TestFeatExtractionModels(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_diff_model_default_params(self):
        model = DiffFeatureDetectorParam()
        features_a = torch.randn(10, 1024)
        features_b = torch.randn(10, 1024)
        output = model((features_a, features_b))
        self.assertEqual(output.shape, (10, 1))

    def test_diff_model_custom_params(self):
        model = DiffFeatureDetectorParam(num_hidden=2, hidden_units=256, features_dropout=0.7, hidden_dropout=0.25)
        features_a = torch.randn(10, 1024)
        features_b = torch.randn(10, 1024)
        output = model((features_a, features_b))
        self.assertEqual(output.shape, (10, 1))
