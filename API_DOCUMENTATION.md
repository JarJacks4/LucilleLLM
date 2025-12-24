# LucilleLLM API Documentation

**Base URL:** `https://lucillellm2-286076426888.us-east4.run.app`

Hey team! 👋 Here's a quick guide to our chat API endpoints. Let me know if you have questions!

---

## Quick Overview

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/` | GET | Creates a new chat session |
| `/health` | GET | Check if the server is running |
| `/chat` | POST | Send a message, get a complete response |
| `/chat/stream` | POST | Send a message, get response word-by-word (streaming) |

---

## 1. Health Check

**GET** `/health`

Just checks if everything is working.

```bash
curl https://lucillellm2-286076426888.us-east4.run.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "firebase": "connected",
  "vector_store": "loaded"
}
```

---

## 2. Create New Session

**GET** `/`

Creates a fresh chat session. Use this when a user opens the app for the first time.

```bash
curl https://lucillellm2-286076426888.us-east4.run.app/
```

**Response:**
```json
{
  "message": "Welcome to Lucille Self-Care Chat API",
  "session_id": "abc123-def456-ghi789",
  "status": "ready"
}
```

Save this `session_id` - you'll need it for chat requests!

---

## 3. Regular Chat (Recommended for FlutterFlow)

**POST** `/chat`

Send a message and get the full response at once. Best for mobile apps.

### Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "message": "Give me a quick tip for relaxation",
  "session_id": "your-session-id-here"
}
```

### Test in Postman

1. Set method to **POST**
2. URL: `https://lucillellm2-286076426888.us-east4.run.app/chat`
3. Go to **Body** → **raw** → **JSON**
4. Paste this:
```json
{
  "message": "How can I reduce stress?",
  "session_id": "test-session-001"
}
```
5. Hit **Send**

### Response
```json
{
  "session_id": "test-session-001",
  "response": "Here are some tips to reduce stress:\n\n1. **Deep Breathing** - Take slow, deep breaths...",
  "conversation": [
    "How can I reduce stress?",
    "Here are some tips to reduce stress:\n\n1. **Deep Breathing** - Take slow, deep breaths..."
  ],
  "status": "success",
  "timestamp": "2025-12-15T23:20:33.468338",
  "message_count": 2
}
```

**What each field means:**
- `response` - The AI's reply (this is what you show to the user)
- `conversation` - Array with [user message, AI response]
- `message_count` - Total messages in this session
- `status` - "success" if everything worked

---

## 4. Streaming Chat (For Real-time "Typing" Effect)

**POST** `/chat/stream`

This is the cool one! 🚀 The response comes word-by-word, so you can show a "typing" animation like ChatGPT.

### How Streaming Works

Instead of waiting for the full response, you get small pieces called **tokens** as they're generated:

```
User sends: "Hi"

Server sends back (one line at a time):
→ data: {"delta": "Hello", "type": "content", ...}
→ data: {"delta": "!", "type": "content", ...}
→ data: {"delta": " How", "type": "content", ...}
→ data: {"delta": " can", "type": "content", ...}
→ data: {"delta": " I", "type": "content", ...}
→ data: {"delta": " help", "type": "content", ...}
→ data: {"delta": " you", "type": "content", ...}
→ data: {"delta": "?", "type": "content", ...}
→ data: {"type": "done", "response": "Hello! How can I help you?", ...}
→ data: [DONE]
```

### Request

Same as `/chat`:

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "message": "Give me a quick tip for relaxation",
  "session_id": "your-session-id-here"
}
```

### Test in Postman

1. Set method to **POST**
2. URL: `https://lucillellm2-286076426888.us-east4.run.app/chat/stream`
3. Go to **Body** → **raw** → **JSON**
4. Paste:
```json
{
  "message": "What is mindfulness?",
  "session_id": "stream-test-001"
}
```
5. Hit **Send**

You'll see the response come in chunks!

### Response Format

**During streaming** - You get multiple events with `type: "content"`:
```json
{"delta": "Mindfulness", "session_id": "stream-test-001", "type": "content"}
{"delta": " is", "session_id": "stream-test-001", "type": "content"}
{"delta": " the", "session_id": "stream-test-001", "type": "content"}
...
```

**At the end** - You get a final event with `type: "done"` containing the complete response:
```json
{
  "type": "done",
  "session_id": "stream-test-001",
  "response": "Mindfulness is the practice of being fully present...",
  "conversation": [
    "What is mindfulness?",
    "Mindfulness is the practice of being fully present..."
  ],
  "status": "success",
  "message_count": 2,
  "timestamp": "2025-12-15T23:20:37.068311"
}
```

**Final signal:**
```
data: [DONE]
```

### What each field means

| Field | Description |
|-------|-------------|
| `delta` | A small piece of the response (word or punctuation) |
| `type` | Either `"content"` (streaming) or `"done"` (finished) |
| `response` | The complete AI response (only in the `done` event) |
| `conversation` | [user message, AI response] |
| `session_id` | Your session ID |
| `message_count` | Total messages in this session |

---

## Test Data (Copy & Paste)

### For `/chat`
```json
{
  "message": "How do I practice deep breathing?",
  "session_id": "demo-session-123"
}
```

### For `/chat/stream`
```json
{
  "message": "Give me 3 quick self-care tips",
  "session_id": "demo-stream-456"
}
```

### More test messages to try:
- `"I'm feeling stressed, what should I do?"`
- `"What is the Inner Smile meditation technique?"`
- `"How can I sleep better tonight?"`
- `"Give me a quick relaxation exercise"`

---

## Which Endpoint Should I Use?

| Use Case | Recommended Endpoint |
|----------|---------------------|
| FlutterFlow app | `/chat` |
| React/Next.js web app | `/chat/stream` |
| Mobile app (Flutter/React Native) | `/chat` (unless you add custom streaming code) |
| Testing in Postman | Either works! |
| ChatGPT-like typing effect | `/chat/stream` |

---

## Error Handling

If something goes wrong, you'll get:

```json
{
  "detail": "Error message here"
}
```

Common issues:
- **Missing session_id** - Make sure to include it in the request body
- **Empty message** - The message field can't be empty
- **Server error** - Check `/health` endpoint

---

## Questions?

Reach out to the team if you run into any issues! 🙌


