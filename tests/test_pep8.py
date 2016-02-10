import unittest2 as unittest
import pep8
from glob import glob


class TestPEP8(unittest.TestCase):
    def test_pep8_conformance(self):
        """Test PEP8 conformance."""
        pep8style = pep8.StyleGuide()
        result = pep8style.check_files(glob('**/*.py') +
                                       glob('*.py'))
        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings)")
