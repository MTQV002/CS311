# 🚀 Groq Models Reference (Free & Fast!)

## Top Models Available (December 2025)

### 1. **Llama 3.3 70B Versatile** ⭐ RECOMMENDED
```env
LLM_MODEL_GROQ=llama-3.3-70b-versatile
```
- **Best for**: General purpose, Vietnamese QA
- **Context**: 128K tokens
- **Speed**: ~400 tokens/sec
- **Free tier**: ✅ Yes

### 2. **Llama 3.1 70B Versatile**
```env
LLM_MODEL_GROQ=llama-3.1-70b-versatile
```
- **Best for**: Complex reasoning
- **Context**: 128K tokens
- **Speed**: ~350 tokens/sec

### 3. **Mixtral 8x7B**
```env
LLM_MODEL_GROQ=mixtral-8x7b-32768
```
- **Best for**: Fast responses
- **Context**: 32K tokens
- **Speed**: ~500 tokens/sec

### 4. **Llama 3.1 8B Instant**
```env
LLM_MODEL_GROQ=llama-3.1-8b-instant
```
- **Best for**: Ultra-fast simple queries
- **Context**: 128K tokens
- **Speed**: ~800 tokens/sec

### 5. **Gemma 2 9B**
```env
LLM_MODEL_GROQ=gemma2-9b-it
```
- **Best for**: Efficient on-device
- **Context**: 8K tokens
- **Speed**: ~600 tokens/sec

## How to Use

1. Get FREE API key: https://console.groq.com
2. Update `.env`:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```
3. Restart backend!

## Performance Comparison

| Model | Speed | Context | Quality | Vietnamese Support |
|-------|-------|---------|---------|-------------------|
| Llama 3.3 70B | ⚡⚡⚡ | 128K | ⭐⭐⭐⭐⭐ | ✅ Excellent |
| Llama 3.1 70B | ⚡⚡⚡ | 128K | ⭐⭐⭐⭐ | ✅ Excellent |
| Mixtral 8x7B | ⚡⚡⚡⚡ | 32K | ⭐⭐⭐⭐ | ✅ Good |
| Llama 3.1 8B | ⚡⚡⚡⚡⚡ | 128K | ⭐⭐⭐ | ✅ Good |

**Current default**: `llama-3.3-70b-versatile` (Best balance)
