import unittest
from demo import execute, category, change_pair

class TestTPEDNAEmbedding(unittest.TestCase):
    def test_five_categories(self):
        for pair, expected in [
            ((120,120),"EZ"), ((185,182),"EN1"), ((225,200),"EN2"),
            ((250,10),"CN"), ((0,255),"NC")]:
            self.assertEqual(category(*pair), expected)

    def test_embed_examples(self):
        self.assertEqual(change_pair(185,182,0,"DE"), (186,180))
        self.assertEqual(change_pair(225,200,0,"LSB"), (224,200))
        self.assertEqual(change_pair(250,10,0,"LSB"), (250,10))

    def test_8_seeds_reversible(self):
        for seed in (20261110,1,7,42,891,2025,123456,919191):
            r=execute(seed)
            self.assertEqual(r["raw_bits"],220)
            self.assertEqual(r["embedded_bits"],32)
            self.assertGreater(r["modified_pixel_count"],40)
            self.assertGreaterEqual(r["original_unique_values"],80)
            self.assertTrue(r["recovery_equal_original"])

if __name__=="__main__":
    unittest.main()
