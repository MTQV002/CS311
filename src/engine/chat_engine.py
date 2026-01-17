from typing import List, Optional, AsyncGenerator, Tuple
from enum import Enum
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import TextNode, NodeWithScore

from src.config import settings
from src.engine.components import get_llm, get_embed_model, get_reranker, get_vector_store
from src.engine.retriever import HybridRetrieverFactory


# =============================================================================
# INTENT TYPES
# =============================================================================

class IntentType(str, Enum):
    CHAT = "CHAT"  # Chào hỏi, hỏi chung
    LAW = "LAW"    # Câu hỏi pháp luật 


@dataclass
class RouterResult:
    intent: IntentType
    confidence: float
    reasoning: str


# =============================================================================
# PROMPTS
# =============================================================================

# Prompt phân loại intent
ROUTER_PROMPT = """Bạn là bộ phân loại ý định. Xác định câu hỏi thuộc loại nào:

**LAW** - Pháp luật lao động Việt Nam:
- Hợp đồng, lương, thưởng, thai sản, nghỉ phép, sa thải, trợ cấp
- BHXH, BHTN, BHYT, hưu trí, bảo hiểm
- Luật Lao động, Nghị định, điều khoản, quy định
- Ví dụ: "nghỉ thai sản bao lâu?", "trợ cấp thôi việc tính sao?", "BHXH là gì?"

**CHAT** - Chào hỏi, câu hỏi chung không liên quan luật:
- Ví dụ: "xin chào", "bạn là ai?", "cảm ơn", "tạm biệt"

QUAN TRỌNG: Nếu lịch sử có pháp luật và câu hiện tại là follow-up → LAW

Lịch sử: {chat_history}
Câu hỏi: {query}

INTENT: [LAW/CHAT]
CONFIDENCE: [0.0-1.0]
REASONING: [1 dòng]"""

# Prompt rewrite câu hỏi follow-up thành câu độc lập
CONDENSE_PROMPT = """Cho lịch sử và câu hỏi tiếp theo, viết lại thành câu hỏi độc lập.

QUY TẮC QUAN TRỌNG:
1. PHẢI GIỮ NGUYÊN các từ khóa pháp lý: sáp nhập, tái cơ cấu, mang thai, thai sản, nghỉ hưu, sa thải, độc hại, BHTN, BHXH, trợ cấp, hợp đồng
2. PHẢI GIỮ NGUYÊN các con số: số năm làm việc, số tiền lương, tuổi, thời gian đóng bảo hiểm
3. PHẢI GIỮ NGUYÊN lý do nghỉ việc nếu có đề cập
4. Chỉ viết lại để câu hỏi rõ ràng hơn, KHÔNG THAY ĐỔI ý nghĩa

Lịch sử: {chat_history}
Câu hỏi: {question}

Câu hỏi đã viết lại:"""

# Prompt hướng dẫn LLM trả lời dựa trên context pháp luật
CONTEXT_PROMPT = """Bạn là trợ lý AI chuyên gia về Pháp luật Lao động Việt Nam.

CƠ SỞ DỮ LIỆU: Bộ luật Lao động 2019, Luật ATVSLĐ 2015, Luật BHXH 2024, Luật Việc làm 2024, NĐ145/2020, NĐ12/2022, NĐ293/2025

NGUYÊN TẮC TRẢ LỜI:
1. Chỉ dựa vào điều khoản được cung cấp
2. Trích dẫn tên văn bản, số Điều, Khoản
3. Trả lời tiếng Việt, rõ ràng, ngắn gọn
4. Nếu không có thông tin, trả lời 'Câu hỏi của bạn không nằm trong phạm vi của tôi.'

QUY TẮC TÍNH TOÁN TRỢ CẤP (nếu hỏi về trợ cấp):
- Trợ cấp THÔI VIỆC (Điều 46): 0.5 tháng lương × số năm = khi tự nghỉ, hết hạn HĐ
- Trợ cấp MẤT VIỆC LÀM (Điều 47): 1 tháng lương × số năm, tối thiểu 2 tháng = khi sáp nhập, tái cơ cấu, cắt giảm
- Thời gian tính = Tổng thời gian làm việc - Thời gian đóng BHTN
- Làm tròn: dưới 6 tháng → 0.5 năm, từ 6 tháng → 1 năm
- Trợ cấp thất nghiệp (Điều 50 Luật VL): 60% lương × số tháng (mỗi 12 tháng đóng = 3 tháng hưởng)

CÁC ĐIỀU KHOẢN:
{context_str}"""

# Prompt cho CHAT intent (không cần RAG)
CHAT_RESPONSE_PROMPT = """Bạn là trợ lý AI thân thiện chuyên về Pháp luật Lao động Việt Nam.

NHIỆM VỤ: Trả lời chào hỏi, giới thiệu bản thân, hoặc hướng dẫn người dùng.

VÍ DỤ:
- "Xin chào" → "Xin chào! Tôi là trợ lý AI về pháp luật lao động. Bạn cần hỏi gì?"
- "Bạn là ai?" → "Tôi có thể giúp về Bộ luật Lao động, BHXH, hợp đồng, tiền lương..."

Lịch sử: {chat_history} 
Câu hỏi: {query}
Trả lời:"""


# =============================================================================
# SEMANTIC ROUTER - Phân loại intent LAW/CHAT
# =============================================================================

class SemanticRouter:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
    
    def _parse_response(self, response_text: str) -> RouterResult:
        try:
            lines = response_text.strip().split('\n')
            intent_line = next((l for l in lines if l.startswith('INTENT:')), None)
            confidence_line = next((l for l in lines if l.startswith('CONFIDENCE:')), None)
            reasoning_line = next((l for l in lines if l.startswith('REASONING:')), None)
            
            intent_str = intent_line.split(':')[1].strip().upper() if intent_line else "LAW"
            confidence = float(confidence_line.split(':')[1].strip()) if confidence_line else 0.8
            reasoning = reasoning_line.split(':', 1)[1].strip() if reasoning_line else ""
            
            return RouterResult(
                intent=IntentType.LAW if intent_str == "LAW" else IntentType.CHAT,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception:
            return RouterResult(intent=IntentType.LAW, confidence=0.5, reasoning="Parse error")
    
    def route(self, query: str, chat_history: str = "") -> RouterResult:
        """Sync routing"""
        prompt = ROUTER_PROMPT.format(query=query, chat_history=chat_history or "(Chưa có)")
        response = self.llm.complete(prompt)
        return self._parse_response(str(response))
    
    async def aroute(self, query: str, chat_history: str = "") -> RouterResult:
        """Async routing"""
        prompt = ROUTER_PROMPT.format(query=query, chat_history=chat_history or "(Chưa có)")
        response = await self.llm.acomplete(prompt)
        return self._parse_response(str(response))


# =============================================================================
# CHAT ENGINE MANAGER - Quản lý toàn bộ RAG pipeline
# =============================================================================

class ChatEngineManager:
    """
    Luồng xử lý:
    1. Router phân loại intent (LAW/CHAT)
    2. Nếu CHAT → LLM trả lời trực tiếp
    3. Nếu LAW → Hybrid Search → Rerank → LLM generate với context
    """
    
    def __init__(self, nodes: Optional[List[TextNode]] = None):
        self.nodes = nodes or []
        self.memory_token_limit = settings.MEMORY_TOKEN_LIMIT        
        self.llm = None
        self.embed_model = None
        self.reranker = None
        self.router = None
        self.hybrid_retriever = None
        self.chat_engine = None
        self.memory = None
        self.vector_store = None
        self._initialized = False
    
    def initialize(self, nodes: Optional[List[TextNode]] = None):
        """Khởi tạo tất cả components"""
        if nodes:
            self.nodes = nodes
        
        print("🚀 Initializing Chat Engine Manager...")
        
        # Load models
        print("[1/6] Loading LLM...")
        self.llm = get_llm()
        
        print("[2/6] Loading embedding model...")
        self.embed_model = get_embed_model()
        
        print("[3/6] Loading reranker...")
        self.reranker = get_reranker()
        
        # Router để phân loại intent
        print("[4/6] Initializing semantic router...")
        self.router = SemanticRouter(self.llm)
        
        # Vector store + Hybrid retriever
        print("[5/6] Creating vector index and hybrid retriever...")
        self.vector_store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )
        
        if self.nodes:
            # Hybrid = Vector + BM25 + RRF fusion
            self.hybrid_retriever = HybridRetrieverFactory.create_from_index(index=index, nodes=self.nodes)
        else:
            print("⚠️ No nodes provided, using vector-only retrieval")
            self.hybrid_retriever = index.as_retriever(similarity_top_k=settings.VECTOR_TOP_K)
        
        # Chat engine với memory
        print("[6/6] Creating chat engine with memory...")
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=self.memory_token_limit)
        
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.hybrid_retriever,
            llm=self.llm,
            memory=self.memory,
            node_postprocessors=[self.reranker] if self.reranker else None,
            context_prompt=CONTEXT_PROMPT,
            condense_prompt=CONDENSE_PROMPT,
            verbose=True,
        )
        
        self._initialized = True
        print("✅ Chat Engine Manager initialized successfully!")
    
    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError("ChatEngineManager not initialized. Call initialize() first.")
    
    def reset(self):
        """Reset conversation memory"""
        self._ensure_initialized()
        self.memory.reset()
    
    def _get_recent_history(self, max_turns: int = 5) -> str:
        """Lấy lịch sử gần đây để router có context"""
        try:
            messages = self.memory.get_all()
            if not messages:
                return ""
            recent = messages[-(max_turns * 2):]
            lines = []
            for msg in recent:
                role = "Người dùng" if msg.role == MessageRole.USER else "Trợ lý"
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                lines.append(f"{role}: {content}")
            return "\n".join(lines)
        except Exception:
            return ""
    
    def _handle_chat_intent(self, query: str, chat_history: str = "") -> str:
        """Xử lý CHAT intent - không cần RAG"""
        prompt = CHAT_RESPONSE_PROMPT.format(query=query, chat_history=chat_history or "(Chưa có)")
        response = self.llm.complete(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    
    async def _ahandle_chat_intent(self, query: str, chat_history: str = "") -> str:
        """Async version"""
        prompt = CHAT_RESPONSE_PROMPT.format(query=query, chat_history=chat_history or "(Chưa có)")
        response = await self.llm.acomplete(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    
    # =========================================================================
    # MAIN CHAT METHODS
    # =========================================================================
    
    def chat(self, query: str, skip_routing: bool = False) -> Tuple[str, IntentType, List[NodeWithScore]]:
        """
        Sync chat method
        Returns: (response_text, intent, source_nodes)
        """
        self._ensure_initialized()
        source_nodes = []
        
        # Step 1: Route intent
        if skip_routing:
            intent = IntentType.LAW
        else:
            router_result = self.router.route(query, self._get_recent_history())
            intent = router_result.intent
            print(f"🎯 Router: {intent.value} (confidence: {router_result.confidence:.2f})")
        
        # Step 2: Handle based on intent
        if intent == IntentType.CHAT:
            response_text = self._handle_chat_intent(query, self._get_recent_history())
            self.memory.put(ChatMessage(role=MessageRole.USER, content=query))
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text or ""))
        else:
            # LAW intent → Use RAG chat engine
            response = self.chat_engine.chat(query)
            response_text = str(response) if response else ""
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        return response_text or "Xin lỗi, tôi không thể tạo câu trả lời.", intent, source_nodes
    
    async def achat(self, query: str, skip_routing: bool = False) -> Tuple[str, IntentType, List[NodeWithScore]]:
        """Async chat method"""
        self._ensure_initialized()
        source_nodes = []
        
        if skip_routing:
            intent = IntentType.LAW
        else:
            router_result = await self.router.aroute(query, self._get_recent_history())
            intent = router_result.intent
            print(f"🎯 Router: {intent.value} (confidence: {router_result.confidence:.2f})")

        if intent == IntentType.CHAT:
            response_text = await self._ahandle_chat_intent(query, self._get_recent_history())
            self.memory.put(ChatMessage(role=MessageRole.USER, content=query))
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text or ""))
        else:
            response = await self.chat_engine.achat(query)
            response_text = str(response) if response else ""
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        return response_text or "Xin lỗi, tôi không thể tạo câu trả lời.", intent, source_nodes
    
    async def astream_chat(
        self, query: str, skip_routing: bool = False
    ) -> AsyncGenerator[Tuple[str, Optional[IntentType], Optional[List[NodeWithScore]]], None]:
        """
        Streaming chat - yield từng chunk text
        Cuối cùng yield intent và source_nodes
        """
        self._ensure_initialized()
        
        # Route intent
        if skip_routing:
            intent = IntentType.LAW
        else:
            router_result = await self.router.aroute(query, self._get_recent_history())
            intent = router_result.intent
            print(f"🎯 Router: {intent.value} (confidence: {router_result.confidence:.2f})")
        
        if intent == IntentType.CHAT:
            # Stream từ LLM trực tiếp
            chat_history = self._get_recent_history() or "(Chưa có)"
            prompt = CHAT_RESPONSE_PROMPT.format(query=query, chat_history=chat_history)
            full_response = ""
            
            async for chunk in await self.llm.astream_complete(prompt):
                chunk_text = chunk.delta if hasattr(chunk, 'delta') else str(chunk)
                full_response += chunk_text
                yield chunk_text, None, None
            
            self.memory.put(ChatMessage(role=MessageRole.USER, content=query))
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=full_response))
            yield "", intent, []
        else:
            # Stream từ RAG chat engine
            streaming_response = await self.chat_engine.astream_chat(query)
            source_nodes = []
            
            async for chunk in streaming_response.async_response_gen():
                yield chunk, None, None
            
            if hasattr(streaming_response, 'source_nodes'):
                source_nodes = streaming_response.source_nodes
            yield "", intent, source_nodes


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_chat_engine_manager: Optional[ChatEngineManager] = None


def get_chat_engine_manager() -> ChatEngineManager:
    global _chat_engine_manager
    if _chat_engine_manager is None:
        _chat_engine_manager = ChatEngineManager()
    return _chat_engine_manager


def set_chat_engine_manager(manager: ChatEngineManager):
    global _chat_engine_manager
    _chat_engine_manager = manager
