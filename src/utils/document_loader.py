"""Plain-text document loaders for files and directory trees."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


class TextFileLoader:
    """Load one text file into a LangChain ``Document``."""

    def __init__(
        self,
        file_path: str | Path,
        *,
        encoding: str = "utf-8",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.file_path = Path(file_path).expanduser().resolve()
        self.encoding = encoding
        self._metadata = dict(metadata or {})

    def load(self) -> list[Document]:
        """Read the file and return it as a one-item document list."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Text file does not exist: {self.file_path}")
        if not self.file_path.is_file():
            raise IsADirectoryError(f"Expected a text file: {self.file_path}")

        metadata = {
            **self._metadata,
            "source": str(self.file_path),
            "file_name": self.file_path.name,
        }
        return [
            Document(
                page_content=self.file_path.read_text(encoding=self.encoding),
                metadata=metadata,
            )
        ]


class TextDirectoryLoader:
    """Load matching text files from a directory in deterministic order."""

    def __init__(
        self,
        directory_path: str | Path,
        *,
        glob_pattern: str = "*.txt",
        recursive: bool = True,
        encoding: str = "utf-8",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not glob_pattern.strip():
            raise ValueError("glob_pattern cannot be empty")

        self.directory_path = Path(directory_path).expanduser().resolve()
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.encoding = encoding
        self._metadata = dict(metadata or {})

    def load(self) -> list[Document]:
        """Load all matching files, returning an empty list when none match."""
        if not self.directory_path.exists():
            raise FileNotFoundError(
                f"Text directory does not exist: {self.directory_path}"
            )
        if not self.directory_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {self.directory_path}"
            )

        candidates = (
            self.directory_path.rglob(self.glob_pattern)
            if self.recursive
            else self.directory_path.glob(self.glob_pattern)
        )
        file_paths = sorted(
            (path for path in candidates if path.is_file()),
            key=lambda path: path.relative_to(self.directory_path)
            .as_posix()
            .casefold(),
        )

        documents: list[Document] = []
        for file_path in file_paths:
            relative_path = file_path.relative_to(self.directory_path).as_posix()
            documents.extend(
                TextFileLoader(
                    file_path,
                    encoding=self.encoding,
                    metadata={
                        **self._metadata,
                        "relative_path": relative_path,
                    },
                ).load()
            )
        return documents


# Compatibility spelling for callers that use "Textfile" as one word.
TextfileLoader = TextFileLoader


__all__ = ["TextDirectoryLoader", "TextFileLoader", "TextfileLoader"]
