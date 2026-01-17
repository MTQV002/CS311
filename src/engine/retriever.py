"""
Hybrid Retriever - Vector + BM25 with RRF Fusion
"""
from typing import List
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever

from src.config import settings


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever combining Dense Vector Search and BM25 with RRF fusion."""
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        top_k: int = 15,
        rrf_k: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self.vector_retriever.retrieve, query_bundle)
            bm25_future = executor.submit(self.bm25_retriever.retrieve, query_bundle)
            
            vector_nodes = vector_future.result()
            bm25_nodes = bm25_future.result()
        
        fused_nodes = self._rrf_fusion(vector_nodes, bm25_nodes)
        return fused_nodes[:self.top_k]
    
    def _rrf_fusion(
        self,
        vector_nodes: List[NodeWithScore],
        bm25_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        rrf_scores = {}
        node_map = {}
        
        for rank, node in enumerate(vector_nodes, 1):
            node_id = node.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1 / (self.rrf_k + rank)
            node_map[node_id] = node
        
        for rank, node in enumerate(bm25_nodes, 1):
            node_id = node.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1 / (self.rrf_k + rank)
            if node_id not in node_map:
                node_map[node_id] = node
        
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return [
            NodeWithScore(node=node_map[node_id].node, score=rrf_scores[node_id])
            for node_id in sorted_ids
        ]


class HybridRetrieverFactory:
    """Factory for creating HybridRetriever instances."""
    
    @staticmethod
    def create_from_index(
        index: VectorStoreIndex,
        nodes: List[TextNode],
        vector_top_k: int = None,
        bm25_top_k: int = None,
        hybrid_top_k: int = None,
        rrf_k: int = None,
    ) -> HybridRetriever:
        vector_top_k = vector_top_k or settings.VECTOR_TOP_K
        bm25_top_k = bm25_top_k or settings.BM25_TOP_K
        hybrid_top_k = hybrid_top_k or settings.HYBRID_TOP_K
        rrf_k = rrf_k or settings.RRF_K
        
        vector_retriever = index.as_retriever(similarity_top_k=vector_top_k)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=bm25_top_k)
        
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            top_k=hybrid_top_k,
            rrf_k=rrf_k
        )
