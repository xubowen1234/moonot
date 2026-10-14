"""Unit checks for the inventory gate; no credentials, Git writes, or network."""
import unittest
from audit_project import production_path, line_counts

class InventoryTests(unittest.TestCase):
    def test_only_substantive_locations_can_count(self):
        self.assertTrue(production_path("moonot.mbt"))
        self.assertTrue(production_path("engine/numerics.mbt"))
        for name in ("examples/main.mbt", "benchmarks/bench.mbt", "cmd/main/main.mbt", "vendor/kernel.mbt",
                     "generated/data.mbt", "_build/engine.mbt", "checks_test.mbt", "checks_wbtest.mbt",
                     "pkg.generated.mbti", "docs/readme.md", "moon.mod", ""):
            self.assertFalse(production_path(name), name)

    def test_physical_count_and_transparency(self):
        self.assertEqual(line_counts("///|\n// comment\n\nfn f {}\n"),
                         {"physical":4, "nonblank":3, "nonblank_noncomment":1})
        self.assertEqual(line_counts(""), {"physical":0, "nonblank":0, "nonblank_noncomment":0})

    def test_line_endings_do_not_inflate(self):
        self.assertEqual(line_counts("a\r\n\r\nb\r\n"), line_counts("a\n\nb\n"))

if __name__ == "__main__":
    unittest.main()
