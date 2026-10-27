"""Publication controls must never modify the scientific handoff or its reference."""
from pathlib import Path
import re
import tempfile
import unittest
from comparison import publish


class ComparisonTests(unittest.TestCase):
    def test_reference_is_added_only_to_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'handoff';source.mkdir()
            html='<html><head></head><body>synthetic report</body></html>'
            (source/'index.html').write_text(html)
            refs=root/'references';refs.mkdir()
            original='<html><body>synthetic reference</body></html>'
            (refs/'L1_N2.html').write_text(original)
            output=root/'published'
            self.assertTrue(publish(source,output,'L1_N2',refs))
            self.assertEqual((source/'index.html').read_text(),html)
            self.assertFalse((source/'comparison').exists())
            self.assertEqual((output/'comparison/original.html').read_text(),original)
            script=re.search(r'src="(comparison/viewer\.[0-9a-f]{12}\.js)"',(output/'index.html').read_text())
            self.assertIsNotNone(script)
            self.assertTrue((output/script[1]).is_file())

    def test_no_reference_still_publishes_independent_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'handoff';source.mkdir()
            html='<html><head></head><body>synthetic report</body></html>'
            (source/'index.html').write_text(html)
            output=root/'published'
            self.assertFalse(publish(source,output,'L1_N2',root/'absent'))
            self.assertEqual((output/'index.html').read_text(),html)
            self.assertFalse((output/'comparison').exists())


if __name__=='__main__':unittest.main()
