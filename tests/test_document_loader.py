"""Tests for plain-text document loaders."""

import tempfile
import unittest
from pathlib import Path

from utils import TextDirectoryLoader, TextFileLoader, TextfileLoader


class TextFileLoaderTests(unittest.TestCase):
    def test_loads_utf8_text_with_canonical_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "sample.txt"
            file_path.write_text("Hello, Chitta!", encoding="utf-8")

            documents = TextFileLoader(
                file_path,
                metadata={"category": "fixture", "source": "ignored"},
            ).load()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "Hello, Chitta!")
            self.assertEqual(documents[0].metadata["category"], "fixture")
            self.assertEqual(documents[0].metadata["source"], str(file_path.resolve()))
            self.assertEqual(documents[0].metadata["file_name"], "sample.txt")

    def test_missing_file_raises_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.txt"

            with self.assertRaises(FileNotFoundError):
                TextFileLoader(missing_path).load()

    def test_directory_path_raises_is_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(IsADirectoryError):
                TextFileLoader(temporary_directory).load()

    def test_invalid_utf8_propagates_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "invalid.txt"
            file_path.write_bytes(b"\xff")

            with self.assertRaises(UnicodeDecodeError):
                TextFileLoader(file_path).load()

    def test_compatibility_spelling_points_to_same_loader(self) -> None:
        self.assertIs(TextfileLoader, TextFileLoader)


class TextDirectoryLoaderTests(unittest.TestCase):
    def test_loads_txt_files_recursively_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "b.txt").write_text("B", encoding="utf-8")
            (root / "a.txt").write_text("A", encoding="utf-8")
            (nested / "c.txt").write_text("C", encoding="utf-8")
            (nested / "ignored.md").write_text("ignored", encoding="utf-8")

            documents = TextDirectoryLoader(
                root,
                metadata={"dataset": "test"},
            ).load()

            self.assertEqual(
                [document.page_content for document in documents],
                ["A", "B", "C"],
            )
            self.assertEqual(
                [document.metadata["relative_path"] for document in documents],
                ["a.txt", "b.txt", "nested/c.txt"],
            )
            self.assertTrue(
                all(document.metadata["dataset"] == "test" for document in documents)
            )

    def test_non_recursive_mode_ignores_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "root.txt").write_text("root", encoding="utf-8")
            (nested / "nested.txt").write_text("nested", encoding="utf-8")

            documents = TextDirectoryLoader(root, recursive=False).load()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["relative_path"], "root.txt")

    def test_empty_directory_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            self.assertEqual(TextDirectoryLoader(temporary_directory).load(), [])

    def test_file_path_raises_not_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "sample.txt"
            file_path.write_text("text", encoding="utf-8")

            with self.assertRaises(NotADirectoryError):
                TextDirectoryLoader(file_path).load()

    def test_empty_glob_pattern_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TextDirectoryLoader(".", glob_pattern=" ")


if __name__ == "__main__":
    unittest.main()
