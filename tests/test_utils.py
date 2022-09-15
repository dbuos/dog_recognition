import unittest

from drecg.data.utils import list_file_pairs, validate_file_pair

class TestCreateDataset(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.pairs = list(list_file_pairs(part='same', root='/home/daniel/data_dogs/testing'))
        self.pairs_train = list(list_file_pairs(part='same', root='/home/daniel/data_dogs/training'))
        self.pairs_valid = list(list_file_pairs(part='same', root='/home/daniel/data_dogs/validation'))
    
    def test_load_file_pairs(self):
        self.assertEqual(len(self.pairs), 173)
        self.assertEqual(len(self.pairs_train), 1943)
        self.assertEqual(len(self.pairs_valid), 82)

    def test_pairs_should_match(self):
        self.validate_pairs(self.pairs)
        self.validate_pairs(self.pairs_train)

    def test_different_and_same_should_have_same_length(self):
        self.assertEqual(len(list(list_file_pairs(part='same', root='/home/daniel/data_dogs/testing'))), len(list(list_file_pairs(part='different', root='/home/daniel/data_dogs/testing'))))    
        self.assertEqual(len(list(list_file_pairs(part='same', root='/home/daniel/data_dogs/training'))), len(list(list_file_pairs(part='different', root='/home/daniel/data_dogs/training'))))

    def validate_pairs(self, pairs):
        for pair in pairs:
            self.assertTrue(validate_file_pair(pair))      

            
