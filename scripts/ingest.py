#!/usr/bin/env python3
"""
RAG v3 - Ingestion Pipeline (DOCX Version)
==========================================
Chuyên dụng cho file Word (.docx) Bộ luật Lao động.
Ưu điểm:
- Đọc text chính xác 100% (không bị lỗi OCR/Scan).
- Tự động chuẩn hóa Unicode tiếng Việt.
- Tạo ID chuẩn UUID cho Qdrant.
"""
import sys
import os
import argparse
import re
import uuid
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Thư viện đọc file Word
try:
    import docx
except ImportError:
    print("❌ Lỗi: Chưa cài thư viện python-docx.")
    print("👉 Vui lòng chạy: pip install python-docx")
    sys.exit(1)

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config import settings


class VietnamLaborLawDocxParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"📄 Loading DOCX: {file_path}")
        try:
            self.doc = docx.Document(file_path)
        except Exception as e:
            print(f"❌ Không thể mở file Word: {e}")
            sys.exit(1)
        self.full_text = ""
    
    def extract_full_text(self) -> str:
        """Đọc toàn bộ text từ các đoạn văn (paragraph) trong file Word"""
        print("⏳ Extracting text from paragraphs...")
        
        text_parts = []
        for para in tqdm(self.doc.paragraphs, desc="Reading paragraphs"):
            # Chỉ lấy các dòng có nội dung (bỏ dòng trống)
            clean_text = para.text.strip()
            if clean_text:
                text_parts.append(clean_text)
            
        raw_text = "\n".join(text_parts)
            
        # 1. Chuẩn hóa Unicode (Quan trọng cho tiếng Việt: Tổ hợp -> Dựng sẵn)
        self.full_text = unicodedata.normalize('NFKC', raw_text)
        
        # 2. Xử lý khoảng trắng đặc biệt (Non-breaking space)
        self.full_text = self.full_text.replace('\xa0', ' ')
        
        # 3. Xử lý xuống dòng thừa
        self.full_text = re.sub(r'\n{3,}', '\n\n', self.full_text)
        
        return self.full_text
    
    def parse_hierarchical(self) -> List[Dict]:
        """Phân tích cấu trúc Điều/Khoản từ văn bản đã làm sạch"""
        text = self.full_text
        chunks = []
        
        # Regex Patterns (Đã tối ưu cho tiếng Việt)
        chapter_pattern = r'Chương\s+([IVX0-9]+)(?:[:.\s]+([^\n]*))?'
        section_pattern = r'Mục\s+(\d+)(?:[:.\s]+([^\n]*))?'
        article_pattern = r'(?:Điều|ĐIỀU)\s+(\d+)\s*[.:]?\s*(.*?)(?=(?:(?:Điều|ĐIỀU)\s+\d+|Chương\s+[IVX0-9]+|$))'
        
        flags = re.IGNORECASE | re.MULTILINE | re.DOTALL

        # Quét cấu trúc tổng thể
        chapters = [(m.start(), m.group(1), m.group(2).strip() if m.group(2) else "") 
                    for m in re.finditer(chapter_pattern, text, flags)]
        
        # Quét các Điều luật
        articles = list(re.finditer(article_pattern, text, flags))
        
        print(f"📊 Found {len(chapters)} chapters, {len(articles)} articles")
        
        # FALLBACK: Nếu regex thất bại (dù file docx ít khi bị), dùng Sliding Window
        if len(articles) < 5:
            print("⚠️  Cảnh báo: Không tìm thấy đủ cấu trúc Điều luật. Chuyển sang chế độ Cắt Lát (Sliding Window).")
            return self._sliding_window_chunking(text)

        # Phân tích chi tiết từng Điều
        for article_match in tqdm(articles, desc="Parsing articles"):
            article_num = article_match.group(1)
            article_content = article_match.group(2).strip()
            article_pos = article_match.start()
            
            # Tìm chương chứa điều này
            current_chapter = next((c for c in reversed(chapters) if c[0] < article_pos), (None, "?", ""))
            
            # Tách Khoản (1. abc...)
            clause_pattern = r'^(\d+)\.\s+(.+?)(?=(?:^\d+\.\s+|$))'
            clauses = list(re.finditer(clause_pattern, article_content, re.MULTILINE | re.DOTALL))
            
            meta = {
                "article": article_num,
                "chapter": current_chapter[1],
                "chapter_title": current_chapter[2],
                "source": "Vietnam Labor Law 2019 (DOCX)"
            }
            
            if clauses:
                for c_num, c_text in clauses:
                    if len(c_text.strip()) > 5:
                        full_content = (
                            f"Chương {current_chapter[1]}: {current_chapter[2]}\n"
                            f"Điều {article_num}.\n"
                            f"Khoản {c_num}. {c_text.strip()}"
                        )
                        chunk_meta = meta.copy()
                        chunk_meta.update({"clause": c_num, "type": "clause"})
                        chunks.append({"content": full_content, "metadata": chunk_meta})
            else:
                # Điều không có khoản
                full_content = (
                    f"Chương {current_chapter[1]}: {current_chapter[2]}\n"
                    f"Điều {article_num}.\n"
                    f"{article_content}"
                )
                chunk_meta = meta.copy()
                chunk_meta.update({"clause": None, "type": "article"})
                chunks.append({"content": full_content, "metadata": chunk_meta})
        
        return chunks

    def _sliding_window_chunking(self, text: str, chunk_size=1024, overlap=200):
        """Fallback an toàn: Cắt văn bản thành các đoạn chồng lấp"""
        print(f"🔄 Running Sliding Window (Size={chunk_size})...")
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Cố gắng cắt tại dấu xuống dòng để câu không bị gãy
            last_newline = chunk_text.rfind('\n')
            if last_newline != -1 and last_newline > chunk_size * 0.5:
                end = start + last_newline + 1
                chunk_text = text[start:end]
            
            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "content": chunk_text.strip(),
                    "metadata": {
                        "type": "sliding_window",
                        "source": "Vietnam Labor Law 2019 (Fallback)"
                    }
                })
            start = end - overlap
            
        return chunks


def create_nodes_from_chunks(chunks: List[Dict]) -> List[TextNode]:
    """Tạo Node LlamaIndex với ID là UUID chuẩn"""
    nodes = []
    for chunk in chunks:
        node_id = str(uuid.uuid4()) # Tạo UUID ngẫu nhiên
        node = TextNode(
            text=chunk["content"],
            metadata=chunk["metadata"],
            id_=node_id,
            excluded_embed_metadata_keys=["source"],
            excluded_llm_metadata_keys=["source"]
        )
        nodes.append(node)
    return nodes


def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_API_KEY:
        print(f"☁️  Connecting to Qdrant Cloud: {settings.QDRANT_URL}")
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    else:
        print(f"🖥️  Connecting to local Qdrant: {settings.QDRANT_URL}")
        return QdrantClient(url=settings.QDRANT_URL)


def get_embedding_model():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print(f"🔤 Loading embedding model: {settings.EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
        trust_remote_code=True
    )
    print("✅ Embedding model loaded")
    return embed_model


def ingest_to_qdrant(nodes, client, collection_name, embed_model):
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"⚠️  Collection '{collection_name}' exists. Deleting...")
        client.delete_collection(collection_name)
    
    print(f"📦 Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE)
    )
    
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"📥 Ingesting {len(nodes)} nodes into Qdrant...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    return index


def main():
    # Tên file DOCX mặc định
    DEFAULT_DOCX = "Bộ-luật-45-2019-QH14.docx"
    
    parser = argparse.ArgumentParser(description="Ingest Vietnam Labor Law DOCX into Qdrant")
    parser.add_argument(
        "--file",
        type=str,
        default=f"data/{DEFAULT_DOCX}",
        help="Path to the DOCX file"
    )
    args = parser.parse_args()
    
    file_path = Path(args.file)
    # Xử lý đường dẫn tương đối từ project root
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"   Vui lòng copy file '{DEFAULT_DOCX}' vào thư mục data/")
        sys.exit(1)
        
    collection_name = settings.QDRANT_COLLECTION
    
    print("=" * 60)
    print("🚀 RAG v3 - Ingestion Pipeline (Word/DOCX Version)")
    print("=" * 60)
    
    # 1. Parse
    parser = VietnamLaborLawDocxParser(str(file_path))
    parser.extract_full_text()
    chunks = parser.parse_hierarchical()
    
    if not chunks:
        print("❌ CRITICAL ERROR: No content extracted.")
        sys.exit(1)
        
    print(f"✅ Generated {len(chunks)} chunks")
    
    # 2. Create Nodes
    print("\n🔗 Step 2: Creating LlamaIndex nodes...")
    nodes = create_nodes_from_chunks(chunks)
    
    # 3. Embed & Ingest
    print("\n🧠 Step 3: Loading embedding model...")
    embed_model = get_embedding_model()
    
    print("\n🔌 Step 4: Connecting to Qdrant...")
    client = get_qdrant_client()
    
    print("\n📤 Step 5: Ingesting into Qdrant...")
    ingest_to_qdrant(nodes, client, collection_name, embed_model)
    
    print("\n🎉 Ingestion Complete!")

if __name__ == "__main__":
    main()