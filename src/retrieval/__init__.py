"""Retrieval interfaces and implementations."""

from .base import BaseRetriever
from .dpr import Retriever

__all__ = ["BaseRetriever", "Retriever"]
