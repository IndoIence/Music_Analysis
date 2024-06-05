import unittest
from unittest.mock import patch, MagicMock
from wsd import load_model, load_index, predict, input_from_doc, load_artists


class TestWSD(unittest.TestCase):

    def test_input_from_doc(self):
        doc = MagicMock()
        doc.sents = [["This", "is", "sentence", "1"], ["This", "is", "sentence", "2"]]
        doc.text = "This is sentence 1. This is sentence 2."
        expected_output = [
            ("This", "This [unused0] is [unused1] sentence 1."),
            ("is", "This is [unused0] sentence 1 [unused1]."),
            ("sentence", "This is sentence [unused0] 1. This is [unused1] 2."),
            ("1", "This is sentence 1. This is sentence [unused0] 2 [unused1]."),
            ("This", "[unused0] is [unused1] sentence 2."),
            ("is", "This is [unused0] sentence 2 [unused1]."),
            ("sentence", "This is sentence [unused0] 2 [unused1]."),
            ("2", "This is sentence 1. This is sentence 2 [unused0] [unused1]."),
        ]

        result = list(input_from_doc(doc))

        self.assertEqual(result, expected_output)

    @patch("wsd.CONFIG", {"wsd": {"artists": ["artist1", "artist2"]}})
    @patch("wsd.sanitize_art_name")
    @patch("wsd.get_artist")
    def test_load_artists(self, mock_get_artist, mock_sanitize_art_name):
        mock_artist1 = MagicMock()
        mock_artist2 = MagicMock()
        mock_get_artist.side_effect = [mock_artist1, mock_artist2]
        mock_sanitize_art_name.side_effect = ["artist1_sanitized", "artist2_sanitized"]

        artists = list(load_artists())

        mock_sanitize_art_name.assert_any_call("artist1")
        mock_sanitize_art_name.assert_any_call("artist2")
        mock_get_artist.assert_any_call("artist1_sanitized")
        mock_get_artist.assert_any_call("artist2_sanitized")
        self.assertEqual(artists, [mock_artist1, mock_artist2])


if __name__ == "__main__":
    unittest.main()
