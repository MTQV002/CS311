"""
RAG v3 - Custom Hybrid Retriever
=================================
Hybrid retrieval combining:
- Dense Vector Search (Qdrant)
- Sparse Keyword Search (In-memory BM25)
- Reciprocal Rank Fusion (RRF) for result merging
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.config import settings


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever combining Vector Search and BM25.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both
    retrieval methods, providing better recall than either method alone.
    
    Attributes:
        vector_retriever: Dense vector retriever from Qdrant
        bm25_retriever: Sparse keyword retriever (in-memory)
        vector_weight: Weight for vector search results (0-1)
        bm25_weight: Weight for BM25 results (0-1)
        rrf_k: RRF constant (typically 60)
    """
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize HybridRetriever.
        
        Args:
            vector_retriever: Vector store retriever
            bm25_retriever: BM25 retriever
            vector_top_k: Number of results from vector search
            bm25_top_k: Number of results from BM25
            rrf_k: RRF fusion constant
            vector_weight: Weight for vector results in RRF
            bm25_weight: Weight for BM25 results in RRF
        """
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using hybrid approach.
        
        Pipeline:
        1. Vector search -> Top K dense results
        2. BM25 search -> Top K sparse results  
        3. RRF fusion -> Combined ranked results
        
        Args:
            query_bundle: Query with text and embedding
            
        Returns:
            List of NodeWithScore after RRF fusion
        """
        # Step 1: Vector search
        try:
            vector_nodes = self.vector_retriever.retrieve(query_bundle)
            print(f"🔍 Vector search: {len(vector_nodes)} results")
        except Exception as e:
            print(f"⚠️ Vector search error: {e}")
            vector_nodes = []
        
        # Step 2: BM25 search
        try:
            bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
            print(f"🔍 BM25 search: {len(bm25_nodes)} results")
        except Exception as e:
            print(f"⚠️ BM25 search error: {e}")
            bm25_nodes = []
        
        # Handle edge cases
        if not vector_nodes and not bm25_nodes:
            print("⚠️ No results from both retrievers")
            return []
        
        if not vector_nodes:
            return bm25_nodes
        if not bm25_nodes:
            return vector_nodes
        
        # Step 3: RRF fusion
        combined_nodes = self._reciprocal_rank_fusion(vector_nodes, bm25_nodes)
        print(f"🔀 RRF fusion: {len(combined_nodes)} combined results")
        
        return combined_nodes
    
    def _get_node_key(self, node: NodeWithScore) -> str:
        """
        Get unique identifier for a node based on metadata.
        
        Uses article/clause metadata for deduplication when available,
        falls back to node ID or content hash.
        """
        metadata = node.node.metadata or {}
        article = metadata.get('article', '')
        clause = metadata.get('clause', '')
        
        if article and clause:
            return f"article_{article}_clause_{clause}"
        elif article:
            return f"article_{article}"
        else:
            # Fallback to node ID or content hash
            return node.node.id_ or str(hash(node.node.get_content()[:200]))
    
    def _reciprocal_rank_fusion(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF Formula: score = Σ (weight / (k + rank))
        
        This method:
        1. Scores each result by its rank in each list
        2. Combines scores using weighted RRF
        3. Deduplicates by content/metadata
        4. Returns sorted combined results
        
        Args:
            vector_nodes: Results from vector search
            bm25_nodes: Results from BM25 search
            
        Returns:
            Combined and sorted list of NodeWithScore
        """
        # Dictionary to track combined scores and source info
        node_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'score': 0.0,
                'node': None,
                'sources': set(),
                'vector_rank': None,
                'bm25_rank': None
            }
        )
        
        # Process vector search results
        for rank, node in enumerate(vector_nodes, start=1):
            node_key = self._get_node_key(node)
            rrf_score = self.vector_weight / (self.rrf_k + rank)
            
            node_data[node_key]['score'] += rrf_score
            node_data[node_key]['sources'].add('vector')
            node_data[node_key]['vector_rank'] = rank
            
            # Keep the node with higher original score
            if node_data[node_key]['node'] is None:
                node_data[node_key]['node'] = node
            elif node.score and node.score > (node_data[node_key]['node'].score or 0):
                node_data[node_key]['node'] = node
        
        # Process BM25 results
        for rank, node in enumerate(bm25_nodes, start=1):
            node_key = self._get_node_key(node)
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            
            node_data[node_key]['score'] += rrf_score
            node_data[node_key]['sources'].add('bm25')
            node_data[node_key]['bm25_rank'] = rank
            
            # Keep the node with higher original score
            if node_data[node_key]['node'] is None:
                node_data[node_key]['node'] = node
        
        # Build result list with RRF scores
        combined_nodes = []
        for node_key, data in node_data.items():
            if data['node'] is not None:
                # Create new NodeWithScore with RRF score
                combined_node = NodeWithScore(
                    node=data['node'].node,
                    score=data['score']
                )
                # Store fusion metadata
                combined_node.node.metadata = combined_node.node.metadata or {}
                combined_node.node.metadata['_rrf_sources'] = list(data['sources'])
                combined_node.node.metadata['_vector_rank'] = data['vector_rank']
                combined_node.node.metadata['_bm25_rank'] = data['bm25_rank']
                
                combined_nodes.append(combined_node)
        
        # Sort by RRF score (descending)
        combined_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return combined_nodes


class HybridRetrieverFactory:
    """Factory class for creating HybridRetriever instances."""
    
    @staticmethod
    def create_from_index(
        index: VectorStoreIndex,
        nodes: List[TextNode],
        vector_top_k: int = None,
        bm25_top_k: int = None,
        rrf_k: int = None,
    ) -> HybridRetriever:
        """
        Create HybridRetriever from an existing VectorStoreIndex.
        
        Args:
            index: LlamaIndex VectorStoreIndex
            nodes: All document nodes for BM25 index
            vector_top_k: Number of vector results
            bm25_top_k: Number of BM25 results
            rrf_k: RRF constant
            
        Returns:
            Configured HybridRetriever instance
        """
        vector_top_k = vector_top_k or settings.VECTOR_TOP_K
        bm25_top_k = bm25_top_k or settings.BM25_TOP_K
        rrf_k = rrf_k or settings.RRF_K
        
        # Create vector retriever from index
        vector_retriever = index.as_retriever(similarity_top_k=vector_top_k)
        
        # Create BM25 retriever from nodes
        print(f"📚 Building BM25 index from {len(nodes)} nodes...")
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=bm25_top_k,
        )
        print("✅ BM25 index built successfully")
        
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
        )
    
    @staticmethod
    def create_from_qdrant(
        embed_model,
        nodes: List[TextNode],
        vector_top_k: int = None,
        bm25_top_k: int = None,
        rrf_k: int = None,
    ) -> HybridRetriever:
        """
        Create HybridRetriever by connecting to Qdrant.
        
        Args:
            embed_model: Embedding model for vector search
            nodes: All document nodes for BM25 index
            vector_top_k: Number of vector results
            bm25_top_k: Number of BM25 results
            rrf_k: RRF constant
            
        Returns:
            Configured HybridRetriever instance
        """
        from src.engine.components import get_vector_store
        
        vector_top_k = vector_top_k or settings.VECTOR_TOP_K
        bm25_top_k = bm25_top_k or settings.BM25_TOP_K
        rrf_k = rrf_k or settings.RRF_K
        
        # Create vector store and index
        vector_store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        vector_retriever = index.as_retriever(similarity_top_k=vector_top_k)
        
        # Create BM25 retriever
        print(f"📚 Building BM25 index from {len(nodes)} nodes...")
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=bm25_top_k,
        )
        print("✅ BM25 index built successfully")
        
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
        )
