"""
FAISS index builder for fast similarity search.

This module creates and manages FAISS indexes for efficient nearest neighbor
search on item embeddings. Supports both exact and approximate search.

Usage:
    from src.models.index import FAISSIndex

    # Build index from embeddings
    index = FAISSIndex(embedding_dim=64)
    index.build(item_embeddings)

    # Find similar items
    similar = index.search(query_embedding, k=10)

    # Save/load index
    index.save("item_index.faiss")
    index = FAISSIndex.load("item_index.faiss")
"""

import logging
from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """FAISS index wrapper for item similarity search.

    Supports multiple index types:
    - "flat": Exact search using inner product (IndexFlatIP)
    - "ivf": Approximate search using inverted file index (IndexIVFFlat)
    - "hnsw": Approximate search using hierarchical navigable small world graphs

    Attributes:
        embedding_dim: Dimension of the embeddings.
        index_type: Type of FAISS index ("flat", "ivf", "hnsw").
        n_items: Number of items in the index.
        is_built: Whether the index has been built.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        m_hnsw: int = 32,
    ):
        """Initialize the FAISS index.

        Args:
            embedding_dim: Dimension of the embeddings.
            index_type: Type of index - "flat", "ivf", or "hnsw".
            nlist: Number of clusters for IVF index.
            nprobe: Number of clusters to search for IVF index.
            m_hnsw: Number of connections per layer for HNSW index.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.m_hnsw = m_hnsw

        self._index: Optional[faiss.Index] = None
        self._n_items: int = 0
        self.is_built = False

    @property
    def n_items(self) -> int:
        """Number of items in the index."""
        return self._n_items

    def build(
        self,
        embeddings: np.ndarray,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> None:
        """Build the FAISS index from embeddings.

        Args:
            embeddings: Array of shape (n_items, embedding_dim).
            normalize: Whether to L2-normalize embeddings before indexing.
                      Recommended for cosine similarity via inner product.
            show_progress: Whether to log progress.
        """
        n_items, dim = embeddings.shape

        if dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {dim}"
            )

        if show_progress:
            logger.info(
                f"Building FAISS {self.index_type} index for {n_items:,} items "
                f"with {dim} dimensions"
            )

        # Prepare embeddings (must be float32 for FAISS)
        embeddings_f32 = embeddings.astype(np.float32)

        # Normalize for cosine similarity (inner product of normalized = cosine)
        if normalize:
            norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
            embeddings_f32 = embeddings_f32 / (norms + 1e-10)

        # Create index based on type
        if self.index_type == "flat":
            # Exact search using inner product
            self._index = faiss.IndexFlatIP(dim)

        elif self.index_type == "ivf":
            # Approximate search using inverted file
            quantizer = faiss.IndexFlatIP(dim)
            n_clusters = min(
                self.nlist, n_items // 10
            )  # Adjust clusters for small datasets
            self._index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            # IVF requires training
            self._index.train(embeddings_f32)
            self._index.nprobe = self.nprobe

        elif self.index_type == "hnsw":
            # Approximate search using HNSW
            self._index = faiss.IndexHNSWFlat(dim, self.m_hnsw)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add embeddings to index
        self._index.add(embeddings_f32)

        self._n_items = n_items
        self.is_built = True

        if show_progress:
            logger.info(f"FAISS index built with {self._index.ntotal:,} vectors")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        normalize: bool = True,
    ) -> list[tuple[int, float]]:
        """Search for nearest neighbors of a query embedding.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            k: Number of nearest neighbors to return.
            normalize: Whether to normalize the query (should match build setting).

        Returns:
            List of (item_idx, similarity_score) tuples, sorted by similarity.
        """
        if not self.is_built or self._index is None:
            raise ValueError("Index has not been built. Call build() first.")

        # Prepare query (must be float32 and 2D for FAISS)
        query = query_embedding.astype(np.float32).reshape(1, -1)

        if normalize:
            norm = np.linalg.norm(query)
            query = query / (norm + 1e-10)

        # Search
        k = min(k, self._n_items)  # Can't return more than we have
        similarities, indices = self._index.search(query, k)

        # Convert to list of tuples
        results = [
            (int(idx), float(sim))
            for idx, sim in zip(indices[0], similarities[0])
            if idx >= 0  # FAISS returns -1 for missing results
        ]

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        normalize: bool = True,
    ) -> list[list[tuple[int, float]]]:
        """Search for nearest neighbors of multiple queries.

        Args:
            query_embeddings: Query vectors of shape (n_queries, embedding_dim).
            k: Number of nearest neighbors per query.
            normalize: Whether to normalize queries.

        Returns:
            List of results, one per query. Each result is a list of
            (item_idx, similarity_score) tuples.
        """
        if not self.is_built or self._index is None:
            raise ValueError("Index has not been built. Call build() first.")

        # Prepare queries
        queries = query_embeddings.astype(np.float32)

        if normalize:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / (norms + 1e-10)

        # Search
        k = min(k, self._n_items)
        similarities, indices = self._index.search(queries, k)

        # Convert to list of lists of tuples
        results = []
        for i in range(len(queries)):
            query_results = [
                (int(idx), float(sim))
                for idx, sim in zip(indices[i], similarities[i])
                if idx >= 0
            ]
            results.append(query_results)

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save the index to disk.

        Args:
            path: Path to save the index file.
        """
        if not self.is_built or self._index is None:
            raise ValueError("Index has not been built. Call build() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path))
        logger.info(f"FAISS index saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        embedding_dim: Optional[int] = None,
    ) -> "FAISSIndex":
        """Load an index from disk.

        Args:
            path: Path to the index file.
            embedding_dim: Expected embedding dimension (for validation).

        Returns:
            Loaded FAISSIndex instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        index = faiss.read_index(str(path))

        # Create wrapper instance
        dim = index.d
        if embedding_dim is not None and dim != embedding_dim:
            raise ValueError(f"Dimension mismatch: expected {embedding_dim}, got {dim}")

        instance = cls(embedding_dim=dim)
        instance._index = index
        instance._n_items = index.ntotal
        instance.is_built = True

        logger.info(f"FAISS index loaded from {path}: {index.ntotal:,} vectors")

        return instance

    def get_embedding_by_idx(self, idx: int) -> Optional[np.ndarray]:
        """Retrieve the stored embedding for an item.

        Note: Only works for flat indexes. Returns None for other types.

        Args:
            idx: Item index.

        Returns:
            Embedding vector, or None if not retrievable.
        """
        if not self.is_built or self._index is None:
            raise ValueError("Index has not been built.")

        if idx < 0 or idx >= self._n_items:
            raise ValueError(f"Index {idx} out of bounds (0 to {self._n_items - 1})")

        # Only flat indexes support direct reconstruction
        if hasattr(self._index, "reconstruct"):
            try:
                embedding: np.ndarray = self._index.reconstruct(idx)
                return embedding
            except RuntimeError:
                return None

        return None
