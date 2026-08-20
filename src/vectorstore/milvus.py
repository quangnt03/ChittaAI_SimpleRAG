"""Milvus collection building and dense/BM25 hybrid retrieval."""

import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from .base import BaseVectorStore, Embedding


CHUNK_ID_FIELD = "chunk_id"
DOCUMENT_ID_FIELD = "document_id"
TEXT_FIELD = "text"
METADATA_FIELD = "metadata"
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
BM25_FUNCTION = "text_bm25"
MAX_SEARCH_LIMIT = 16_384


class MilvusCollectionSchemaError(RuntimeError):
    """Raised when an existing collection cannot support hybrid retrieval."""


@dataclass(frozen=True, slots=True)
class MilvusIndexConfig:
    """Parameters used by Milvus to build dense, sparse, and scalar indexes."""

    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    def __post_init__(self) -> None:
        if self.hnsw_m < 2:
            raise ValueError("hnsw_m must be at least 2")
        if self.hnsw_ef_construction < 1:
            raise ValueError("hnsw_ef_construction must be at least 1")
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be greater than 0")
        if not 0 <= self.bm25_b <= 1:
            raise ValueError("bm25_b must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class MilvusSearchConfig:
    """Parameters used for ANN recall, BM25 recall, and RRF fusion."""

    hnsw_ef: int = 64
    sparse_drop_ratio: float = 0.0
    candidate_multiplier: int = 4
    rrf_k: int = 60
    consistency_level: str = "Session"

    def __post_init__(self) -> None:
        if self.hnsw_ef < 1:
            raise ValueError("hnsw_ef must be at least 1")
        if not 0 <= self.sparse_drop_ratio < 1:
            raise ValueError("sparse_drop_ratio must be in [0, 1)")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        if not 1 <= self.rrf_k < MAX_SEARCH_LIMIT:
            raise ValueError("rrf_k must be between 1 and 16383")
        if self.consistency_level not in {
            "Strong",
            "Session",
            "Bounded",
            "Eventually",
        }:
            raise ValueError("unsupported Milvus consistency level")


class MilvusIndexBuilder:
    """Build and validate the Milvus collection used by hybrid RAG."""

    def __init__(self, config: MilvusIndexConfig | None = None) -> None:
        self._config = config or MilvusIndexConfig()

    def ensure_collection(
        self,
        client: MilvusClient,
        collection_name: str,
        dimension: int,
    ) -> bool:
        """Create and load the collection, or validate an existing collection.

        Returns ``True`` when a new collection was created.
        """
        if dimension < 1:
            raise ValueError("embedding dimension must be at least 1")

        created = not client.has_collection(collection_name=collection_name)
        if created:
            self._create_collection(client, collection_name, dimension)
        else:
            self._validate_collection(client, collection_name, dimension)

        client.load_collection(collection_name=collection_name)
        return created

    def _create_collection(
        self,
        client: MilvusClient,
        collection_name: str,
        dimension: int,
    ) -> None:
        schema = client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Dense COSINE + BM25 hybrid RAG chunks",
        )
        schema.add_field(
            field_name=CHUNK_ID_FIELD,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=512,
        )
        schema.add_field(
            field_name=DOCUMENT_ID_FIELD,
            datatype=DataType.VARCHAR,
            max_length=512,
        )
        schema.add_field(
            field_name=TEXT_FIELD,
            datatype=DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"type": "standard"},
        )
        schema.add_field(field_name=METADATA_FIELD, datatype=DataType.JSON)
        schema.add_field(
            field_name=DENSE_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=dimension,
        )
        schema.add_field(
            field_name=SPARSE_FIELD,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            Function(
                name=BM25_FUNCTION,
                function_type=FunctionType.BM25,
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=DENSE_FIELD,
            index_name="dense_hnsw_cosine",
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": self._config.hnsw_m,
                "efConstruction": self._config.hnsw_ef_construction,
            },
        )
        index_params.add_index(
            field_name=SPARSE_FIELD,
            index_name="sparse_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": self._config.bm25_k1,
                "bm25_b": self._config.bm25_b,
            },
        )
        index_params.add_index(
            field_name=DOCUMENT_ID_FIELD,
            index_name="document_id_inverted",
            index_type="INVERTED",
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _validate_collection(
        self,
        client: MilvusClient,
        collection_name: str,
        dimension: int,
    ) -> None:
        description = client.describe_collection(collection_name=collection_name)
        fields = {
            str(field.get("name") or field.get("field_name")): field
            for field in description.get("fields", [])
        }
        required_fields = {
            CHUNK_ID_FIELD,
            DOCUMENT_ID_FIELD,
            TEXT_FIELD,
            METADATA_FIELD,
            DENSE_FIELD,
            SPARSE_FIELD,
        }
        missing_fields = sorted(required_fields.difference(fields))

        dense_field = fields.get(DENSE_FIELD, {})
        dense_params = dense_field.get("params") or {}
        actual_dimension = dense_field.get("dim", dense_params.get("dim"))
        dimension_mismatch = (
            actual_dimension is not None and int(actual_dimension) != dimension
        )

        functions = description.get("functions") or []
        bm25_outputs = {
            str(output)
            for function in functions
            if str(function.get("name")) == BM25_FUNCTION
            for output in (
                function.get("output_field_names")
                or function.get("output_fields")
                or []
            )
        }
        missing_bm25 = SPARSE_FIELD not in bm25_outputs

        if missing_fields or dimension_mismatch or missing_bm25:
            reasons: list[str] = []
            if missing_fields:
                reasons.append(f"missing fields: {', '.join(missing_fields)}")
            if dimension_mismatch:
                reasons.append(
                    f"{DENSE_FIELD} dimension is {actual_dimension}, "
                    f"expected {dimension}"
                )
            if missing_bm25:
                reasons.append(f"missing {BM25_FUNCTION} BM25 function")
            detail = "; ".join(reasons)
            raise MilvusCollectionSchemaError(
                f"Milvus collection {collection_name!r} is not hybrid-search "
                f"compatible ({detail}). BM25 must be defined when a collection "
                "is created. Select a new MILVUS_COLLECTION or explicitly rebuild "
                "the existing collection after preserving any required data."
            )


class MilvusVectorStore(BaseVectorStore):
    """Store chunks and search them with dense, sparse, or hybrid retrieval."""

    def __init__(
        self,
        collection_name: str,
        *,
        index_config: MilvusIndexConfig | None = None,
        search_config: MilvusSearchConfig | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        self._collection_name = collection_name
        self._index_builder = MilvusIndexBuilder(index_config)
        self._search_config = search_config or MilvusSearchConfig()
        self._client: MilvusClient | None = None
        self._dimension: int | None = None

    def connect(self, url: str, port: int, password: str | None = None) -> None:
        """Connect to Milvus, using ``password`` as a token when provided."""
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        endpoint = url.rstrip("/")
        if "://" not in endpoint:
            endpoint = f"http://{endpoint}"

        self._client = MilvusClient(
            uri=f"{endpoint}:{port}",
            token=password or "",
        )
        self._dimension = None

    def build_index(self, dimension: int) -> bool:
        """Build/load the Milvus hybrid collection for ``dimension`` vectors."""
        client = self._require_client("build_index")
        created = self._index_builder.ensure_collection(
            client,
            self._collection_name,
            dimension,
        )
        self._dimension = dimension
        return created

    def insert(self, embeddings: list[Embedding]) -> None:
        """Create the collection when needed and insert embedding records."""
        client = self._require_client("insert")
        if not embeddings:
            return

        dimension = len(embeddings[0].vector)
        if dimension == 0:
            raise ValueError("embedding vectors cannot be empty")
        for embedding in embeddings:
            if len(embedding.vector) != dimension:
                raise ValueError("all embedding vectors must have the same dimension")
            if not embedding.document_id or not embedding.chunk_id:
                raise ValueError("document_id and chunk_id are required for insertion")

        if self._dimension is None:
            self.build_index(dimension)
        elif self._dimension != dimension:
            raise ValueError(
                f"embedding dimension changed from {self._dimension} to {dimension}"
            )

        client.insert(
            collection_name=self._collection_name,
            data=[
                {
                    DENSE_FIELD: embedding.vector,
                    DOCUMENT_ID_FIELD: embedding.document_id,
                    CHUNK_ID_FIELD: embedding.chunk_id,
                    TEXT_FIELD: embedding.text,
                    METADATA_FIELD: embedding.metadata,
                }
                for embedding in embeddings
            ],
        )

    def search(self, query: Embedding, top_k: int) -> list[Embedding]:
        """Search Milvus with cosine similarity and return scored records."""
        client = self._require_client("search")
        if not query.vector:
            raise ValueError("query embedding cannot be empty")
        self._validate_top_k(top_k)

        result = client.search(
            collection_name=self._collection_name,
            data=[query.vector],
            anns_field=DENSE_FIELD,
            limit=top_k,
            output_fields=[
                CHUNK_ID_FIELD,
                DENSE_FIELD,
                DOCUMENT_ID_FIELD,
                TEXT_FIELD,
                METADATA_FIELD,
            ],
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": max(self._search_config.hnsw_ef, top_k)},
            },
            consistency_level=self._search_config.consistency_level,
        )
        return self._parse_hits(result, score_kind="cosine")

    def hybrid_search(self, query: Embedding, top_k: int) -> list[Embedding]:
        """Fuse HNSW/COSINE ANN and BM25 candidates with reciprocal rank fusion."""
        client = self._require_client("hybrid_search")
        if not query.vector:
            raise ValueError("query embedding cannot be empty")
        query_text = query.text.strip()
        if not query_text:
            raise ValueError("query text cannot be empty for BM25 search")
        self._validate_top_k(top_k)

        candidate_k = min(
            MAX_SEARCH_LIMIT,
            max(top_k, top_k * self._search_config.candidate_multiplier),
        )
        dense_request = AnnSearchRequest(
            data=[query.vector],
            anns_field=DENSE_FIELD,
            param={
                "metric_type": "COSINE",
                "params": {"ef": max(self._search_config.hnsw_ef, candidate_k)},
            },
            limit=candidate_k,
        )
        sparse_request = AnnSearchRequest(
            data=[query_text],
            anns_field=SPARSE_FIELD,
            param={
                "metric_type": "BM25",
                "params": {
                    "drop_ratio_search": self._search_config.sparse_drop_ratio,
                },
            },
            limit=candidate_k,
        )
        result = client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(k=self._search_config.rrf_k),
            limit=top_k,
            output_fields=[
                CHUNK_ID_FIELD,
                DENSE_FIELD,
                DOCUMENT_ID_FIELD,
                TEXT_FIELD,
                METADATA_FIELD,
            ],
            consistency_level=self._search_config.consistency_level,
        )

        max_rrf_score = 2.0 / (self._search_config.rrf_k + 1)
        return self._parse_hits(
            result,
            score_kind="confidence",
            score_transform=lambda score: min(1.0, max(0.0, score / max_rrf_score)),
            raw_score_metadata_key="milvus_rrf_score",
        )

    def _parse_hits(
        self,
        result: list[list[dict[str, Any]]],
        *,
        score_kind: str,
        score_transform: Callable[[float], float] | None = None,
        raw_score_metadata_key: str | None = None,
    ) -> list[Embedding]:
        if not result:
            return []

        matches: list[Embedding] = []
        for hit in result[0]:
            entity = hit["entity"]
            raw_score = float(hit["distance"])
            metadata = dict(entity.get(METADATA_FIELD) or {})
            if raw_score_metadata_key is not None:
                metadata[raw_score_metadata_key] = raw_score
            matches.append(
                Embedding(
                    vector=list(entity.get(DENSE_FIELD) or []),
                    document_id=str(entity[DOCUMENT_ID_FIELD]),
                    chunk_id=str(entity.get(CHUNK_ID_FIELD) or hit.get("id") or ""),
                    text=str(entity[TEXT_FIELD]),
                    metadata=metadata,
                    score=(
                        raw_score
                        if score_transform is None
                        else float(score_transform(raw_score))
                    ),
                    score_kind=score_kind,
                )
            )
        return matches

    def delete_document(self, document_id: str) -> None:
        """Delete all embeddings associated with ``document_id``."""
        client = self._require_client("delete_document")
        client.delete(
            collection_name=self._collection_name,
            filter=f"document_id == {json.dumps(document_id)}",
        )

    def delete_chunk(self, document_id: str, chunk_id: str) -> None:
        """Delete one chunk, constrained to its parent document."""
        client = self._require_client("delete_chunk")
        client.delete(
            collection_name=self._collection_name,
            filter=(
                f"document_id == {json.dumps(document_id)} and "
                f"chunk_id == {json.dumps(chunk_id)}"
            ),
        )

    def _require_client(self, operation: str) -> MilvusClient:
        if self._client is None:
            raise RuntimeError(f"connect must be called before {operation}")
        return self._client

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not 1 <= top_k <= MAX_SEARCH_LIMIT:
            raise ValueError("top_k must be between 1 and 16384")
