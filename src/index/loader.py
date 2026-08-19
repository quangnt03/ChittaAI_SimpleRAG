from typing import List
from langchain.embeddings.base import Embeddings
from langchain_text_splitters.base import TextSplitter
from langchain_text_splitters import CharacterTextSplitter


class Chunker:
    def __init__(self, embedding_model: Embeddings, chunker: TextSplitter):
        self.__embedding_model = embedding_model
        self.__embedding_model = chunker
