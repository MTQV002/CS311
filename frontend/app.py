"""
RAG v3 - Chainlit Frontend (Production SSE)
===========================================
Full-featured UI with:
- SSE Streaming
- Source citations with Metadata (Article, Clause, etc.)
- Session Management & Memory Reset
- Beautiful formatting
"""
import os
import json
import chainlit as cl
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

@cl.on_chat_start
async def start():
    """Initialize chat session with Welcome Message"""
    
    # Store session info
    cl.user_session.set("session_id", cl.user_session.get("id"))
    
    # 1. Welcome Message
    welcome_msg = """# 🏛️ Hệ thống Tra cứu Luật Lao động (RAG v3)

Xin chào! Tôi là trợ lý AI chuyên về **Bộ luật Lao động Việt Nam 2019**.
Tôi có thể giúp bạn trả lời các câu hỏi pháp lý dựa trên văn bản luật chính thức.

### 💡 Gợi ý câu hỏi:
- *Thời gian thử việc tối đa là bao lâu?*
- *Người lao động được nghỉ bao nhiêu ngày phép năm?*
- *Khi nào được đơn phương chấm dứt hợp đồng?*
- *Tiền lương làm thêm giờ vào ngày nghỉ lễ tính thế nào?*

*(Dữ liệu được trích xuất từ văn bản gốc, có dẫn chứng Điều/Khoản cụ thể)*
"""
    await cl.Message(content=welcome_msg).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages and Stream response from SSE Backend
    """
    # 1. Create an empty message for streaming
    msg = cl.Message(content="")
    await msg.send()
    
    payload = {"content": message.content}
    
    # 2. Call Backend with httpx
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            async with client.stream("POST", f"{BACKEND_URL}/chat", json=payload) as response:
                
                if response.status_code != 200:
                    err_text = await response.aread()
                    msg.content = f"❌ **Lỗi Server ({response.status_code}):**\n{err_text.decode()}"
                    await msg.update()
                    return

                # Local storage for accumulation
                source_nodes = []
                intent = None

                # 3. Process SSE Stream
                async for line in response.aiter_lines():
                    line = line.strip()
                    
                    # Filter for 'data:' lines
                    if not line.startswith("data:"):
                        continue
                    
                    json_str = line[5:].strip()
                    if not json_str or json_str == "[DONE]":
                        continue
                        
                    try:
                        data = json.loads(json_str)
                        
                        # Handle Errors
                        if "error" in data:
                            msg.content += f"\n\n⚠️ **Lỗi:** {data['error']}"
                            await msg.update()
                            continue

                        # A. Stream Text Token
                        if "token" in data:
                            await msg.stream_token(data["token"])
                        
                        # B. Capture Metadata
                        if "intent" in data:
                            intent = data["intent"]
                        if "nodes" in data:
                            source_nodes = data["nodes"]
                            
                    except json.JSONDecodeError:
                        continue
                
                # 4. Display Sources (After stream finishes)
                if source_nodes:
                    elements = []
                    ref_names = []
                    
                    for idx, node in enumerate(source_nodes):
                        # Extract metadata
                        meta = node.get("metadata", {})
                        score = node.get("score", 0)
                        
                        # Format Title: "Điều 5, Khoản 1 (Chapter Title)"
                        article_num = meta.get('article', '?')
                        clause_num = meta.get('clause')
                        
                        ref_name = f"Điều {article_num}"
                        if clause_num:
                            ref_name += f", Khoản {clause_num}"
                            
                        # Format Content for the popup
                        display_content = f"**{ref_name}**\n"
                        if meta.get('article_title'):
                            display_content += f"_{meta['article_title']}_\n"
                        display_content += f"\n> {node.get('text', '')}"
                        
                        # Create Chainlit Text Element
                        elements.append(
                            cl.Text(
                                name=f"Nguồn {idx+1}",
                                content=display_content,
                                display="inline"
                            )
                        )
                        ref_names.append(f"Nguồn {idx+1}")
                    
                    # Attach elements to message
                    msg.elements = elements
                    
                    # Add footer text if it's a legal query
                    if intent == "LAW":
                        ref_str = ", ".join(ref_names)
                        await msg.stream_token(f"\n\n**🔍 Căn cứ pháp lý:** {ref_str}")
                
                await msg.update()

        except Exception as e:
            msg.content = f"❌ **Lỗi kết nối:** {str(e)}"
            await msg.update()


# ============================================================================
# ACTIONS & CALLBACKS
# ============================================================================

@cl.action_callback("reset_memory")
async def on_reset_memory(action: cl.Action):
    """Callback to reset conversation memory via UI button (if used)"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{BACKEND_URL}/reset-memory")
            data = resp.json()
            
        if data.get("success"):
            await cl.Message(content="🧹 **Đã xóa bộ nhớ hội thoại!**").send()
        else:
            await cl.Message(content=f"❌ **Lỗi:** {data.get('message')}").send()
            
    except Exception as e:
        await cl.Message(content=f"❌ **Lỗi kết nối:** {str(e)}").send()

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings update"""
    pass