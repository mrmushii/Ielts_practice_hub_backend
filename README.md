# IELTS Platform — Backend

A Python/FastAPI backend that powers all four IELTS practice modules (Speaking, Listening, Reading, Writing) plus an omni-tutor assistant. It uses Groq Cloud for LLM inference and STT, Microsoft Edge TTS for free speech synthesis, MongoDB for persistence, and LangGraph for stateful AI workflows.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
  - [Core](#core-endpoints)
  - [Speaking](#speaking-endpoints)
  - [Listening](#listening-endpoints)
  - [Reading](#reading-endpoints)
  - [Writing](#writing-endpoints)
  - [Tutor](#tutor-endpoints)
  - [Documents](#documents-endpoints)
- [Agents](#agents)
- [Utilities](#utilities)
- [Data Models](#data-models)
- [Environment Variables](#environment-variables)
- [Getting Started](#getting-started)

---

## Architecture Overview

```
FastAPI Application (main.py)
│
├── CORS: localhost:3000 + production Vercel URL
├── MongoDB lifecycle (connect on startup, close on shutdown)
│
├── /api          → core.py    (chat, STT, TTS, audio serving)
├── /api/speaking → speaking.py
├── /api/writing  → writing.py
├── /api/reading  → reading.py
├── /api/listening→ listening.py
├── /api/tutor    → tutor.py
└── /api/documents→ documents.py
```

**AI Stack:**
| Layer | Technology |
|-------|-----------|
| LLM | Groq Cloud — `meta-llama/llama-4-scout-17b-16e-instruct` |
| STT | Groq Whisper (`whisper-large-v3`) |
| TTS | Microsoft Edge TTS (free, no key required) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | MongoDB Atlas Vector Search |
| Orchestration | LangGraph (stateful agent graphs) |
| Database | MongoDB via Motor (async) |

---

## Project Structure

```
backend/
├── main.py                    # FastAPI app entry point, CORS, router registration
├── requirements.txt
├── runtime.txt                # python-3.11.9
│
├── routers/
│   ├── core.py                # Chat, STT, TTS, audio serving
│   ├── speaking.py            # Speaking test session management
│   ├── listening.py           # Listening test generation
│   ├── reading.py             # Reading passage generation + answer evaluation
│   ├── writing.py             # Essay prompts, grading, OCR, chart generation
│   ├── tutor.py               # Tutor chat + transcription
│   └── documents.py           # PDF upload → vector store ingestion
│
├── agents/
│   ├── speaking_agent.py      # Examiner dialogue, part transitions, feedback
│   ├── listening_agent.py     # Multi-speaker dialogue + comprehension questions
│   ├── reading_agent.py       # Passage generation + RAG answer evaluation
│   ├── writing_agent.py       # Essay grading, OCR, chart/prompt generation
│   └── tutor_agent.py         # Tool-using tutor (search, docs, datetime)
│
├── models/
│   └── session.py             # Pydantic models: ChatMessage, SpeakingSession
│
├── utils/
│   ├── db.py                  # MongoDB connection + FastAPI dependency injection
│   ├── llm.py                 # Groq LLM singleton factory
│   ├── stt.py                 # Groq Whisper transcription helper
│   ├── tts.py                 # Edge TTS synthesis helper
│   └── langgraph_runtime.py   # LangGraph checkpointer (memory or MongoDB)
│
├── audio_cache/               # Generated TTS audio files (.mp3)
├── generated_charts/          # Dynamically generated Task 1 chart images (.png)
└── temp_uploads/              # Temporary uploaded files (cleared after processing)
```

---

## API Reference

All routes are prefixed with `/api`. The health check lives at the root:

```
GET /health → { "status": "ok", "service": "ielts-platform-api" }
```

---

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Single-turn LLM chat with a configurable system prompt |
| `POST` | `/api/stt` | Transcribe an uploaded audio file via Groq Whisper |
| `POST` | `/api/tts` | Synthesize text to MP3 using Edge TTS; returns audio URL |
| `GET`  | `/api/voices` | List available TTS voice options |
| `GET`  | `/api/langgraph-status` | Report LangGraph checkpointer mode and MongoDB config |
| `GET`  | `/api/audio/{filename}` | Serve a cached audio file from `audio_cache/` |

**Request schemas:**

```jsonc
// POST /api/chat
{ "message": "string", "system_prompt": "string" }

// POST /api/tts
{ "text": "string", "voice": "british_female", "rate": "+0%" }
```

---

### Speaking Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/speaking/start` | Create a new speaking session; returns examiner's opening message |
| `POST` | `/api/speaking/respond` | Send candidate text; examiner replies and advances parts |
| `POST` | `/api/speaking/feedback` | Generate final band score and detailed feedback |
| `GET`  | `/api/speaking/audio/{filename}` | Serve examiner audio for a specific session |

**Session lifecycle:**
1. `start` → Part 1 begins (4–5 questions on familiar topics)
2. `respond` repeatedly → auto-transitions to Part 2 (cue card) and Part 3 (abstract discussion)
3. `feedback` → returns Band 0–9 with per-criterion breakdown

Each `respond` response includes:
```jsonc
{
  "examiner_text": "...",
  "audio_url": "/api/speaking/audio/abc123.mp3",
  "part": 2,
  "is_complete": false
}
```

---

### Listening Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/listening/generate` | Generate a multi-speaker dialogue with comprehension questions |

**Response structure:**
```jsonc
{
  "title": "Booking a Library Study Room",
  "dialogue": [
    { "speaker": "A", "text": "...", "audio_url": "..." },
    { "speaker": "B", "text": "...", "audio_url": "..." }
  ],
  "questions": [
    { "type": "mcq", "question": "...", "options": ["A","B","C","D"], "answer": "B" },
    { "type": "fill_blank", "question": "The room costs £____ per hour.", "answer": "5" }
  ]
}
```

Recent scenarios are tracked to prevent topic repetition across generations.

---

### Reading Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/reading/generate` | AI-generate a 300–400 word academic passage + 3 questions |
| `GET`  | `/api/reading/passages` | Return hardcoded sample passages |
| `POST` | `/api/reading/ask` | Evaluate a user answer using RAG retrieval against the passage |

**RAG evaluation flow:**
1. Embed the question → query MongoDB Atlas Vector Search (`ielts_platform.reading_chunks`)
2. Retrieve top-2 matching chunks
3. LLM compares user answer to retrieved context
4. Return `{ is_correct, feedback, retrieved_context }`

---

### Writing Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/writing/prompts` | Return 3 unique Task 1 prompts + 4 unique Task 2 prompts |
| `POST` | `/api/writing/evaluate` | Grade an essay on all 4 IELTS rubrics (0–9 scale) |
| `POST` | `/api/writing/upload_image` | OCR a handwritten essay image via Groq Vision |
| `GET`  | `/api/writing/generated-chart/{filename}` | Serve a dynamically generated Task 1 chart PNG |

**Evaluation response:**
```jsonc
{
  "task_response_score": 7,
  "coherence_score": 6.5,
  "lexical_score": 7,
  "grammar_score": 6,
  "overall_score": 6.6,
  "strengths": ["Clear position stated in introduction", "..."],
  "weaknesses": ["Limited range of cohesive devices", "..."],
  "improved_version": "Here is a sample improved paragraph: ..."
}
```

Task 1 chart prompts include dynamically generated bar, line, and pie chart images saved to `generated_charts/`.

---

### Tutor Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/tutor/chat` | Send a message to the AI tutor; returns response + suggested module actions |
| `POST` | `/api/tutor/transcribe` | Transcribe voice input using Groq Whisper |

**Tutor response:**
```jsonc
{
  "response": "Great question! For Task 2, you should...",
  "actions": [
    {
      "type": "navigate_module",
      "module": "writing",
      "route": "/writing",
      "label": "Open Writing Module",
      "requires_confirmation": true
    }
  ],
  "meta": { "intent": "writing_help", "confidence": 0.9, "reason": "..." }
}
```

The tutor has access to four tools: `current_datetime`, `internet_search`, `google_search_grounding`, and `search_uploaded_documents`.

---

### Documents Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/documents/upload` | Upload a PDF; chunks and embeds it into MongoDB Atlas Vector Search for RAG |

---

## Agents

### `speaking_agent.py`
Manages examiner persona and part transitions using LangGraph. Randomly selects examiner names (Sarah, Emma, Daniel, Oliver, Hannah, James) and opening styles (warm, formal, energetic, friendly). Automatically advances through Parts 1→2→3 based on conversation history length, then generates final band-score feedback.

### `listening_agent.py`
Generates realistic 10–15 line two-speaker dialogues on IELTS-common topics (bookings, university admin, travel, etc.) and synthesizes each line as separate audio using Edge TTS with voice rotation between speakers. Produces MCQ and fill-in-the-blank comprehension questions with intentional IELTS-style distractors.

### `reading_agent.py`
Two responsibilities:
- **Generation**: creates 300–400 word academic passages (Paleontology, Marine Biology, Psychology, etc.) with 3 mixed-type questions.
- **Evaluation**: a LangGraph RAG pipeline that embeds the question, retrieves relevant passage chunks from MongoDB, and uses the LLM to assess correctness with line-level evidence.

### `writing_agent.py`
Grades essays using structured Pydantic output (exact JSON schema enforced). Also handles:
- **OCR**: Groq Vision API extracts text from handwritten essay photos
- **Chart generation**: PIL-drawn bar/line/pie charts with captions for Task 1 prompts
- **Prompt diversity**: Templates with randomised data to minimise repetition

### `tutor_agent.py`
A LangGraph tool-calling agent with four tools. Maintains session IDs for multi-turn memory. Accepts optional `essay_context` to provide inline writing assistance. Falls back from Google Custom Search to DuckDuckGo if the Google API key is not configured.

---

## Utilities

| File | Purpose |
|------|---------|
| `utils/db.py` | Async Motor client; `get_db()` FastAPI dependency; falls back to `localhost:27017` |
| `utils/llm.py` | Returns a cached `ChatGroq` instance (Llama-4, temp=0.7, max_tokens=2048) |
| `utils/stt.py` | `transcribe_audio(path)` → Groq Whisper; returns `{text, duration}` |
| `utils/tts.py` | `synthesize_speech(text, voice, rate)` → saves MP3 to `audio_cache/`, returns filename |
| `utils/langgraph_runtime.py` | Provides `MemorySaver` or `AsyncMongoDBSaver` based on `LANGGRAPH_CHECKPOINTER` env var |

**Available TTS voices:**

| Key | Voice |
|-----|-------|
| `british_female` (default) | en-GB-SoniaNeural |
| `british_male` | en-GB-RyanNeural |
| `american_female` | en-US-JennyNeural |
| `american_male` | en-US-GuyNeural |
| `australian_female` | en-AU-NatashaNeural |
| `australian_male` | en-AU-WilliamNeural |

---

## Data Models

Defined in `models/session.py`:

```python
class ChatMessage(BaseModel):
    role: str            # "examiner" | "candidate"
    content: str
    timestamp: datetime

class SpeakingSession(BaseModel):
    session_id: str
    part: int            # 1, 2, or 3
    voice: str
    topic_seed: str
    history: List[ChatMessage]
    candidate_name: Optional[str]
    candidate_profile: List[str]
    context_memory: List[str]
    examiner_name: str
    opener_style: str
    feedback: Optional[str]
    is_complete: bool
    created_at: datetime
    updated_at: datetime
```

Sessions are persisted to the `speaking_sessions` MongoDB collection.

---

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Required
GROQ_API_KEY=gsk_...
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/

# Optional — Web search in tutor (falls back to DuckDuckGo if not set)
GOOGLE_SEARCH_API_KEY=
GOOGLE_SEARCH_ENGINE_ID=

# LangGraph checkpointing (default: memory)
LANGGRAPH_CHECKPOINTER=memory          # "memory" | "mongodb"
LANGGRAPH_MONGO_DB=ielts_platform
LANGGRAPH_MONGO_COLLECTION=langgraph_checkpoints

# Feature flags (all default to true)
ENABLE_LANGGRAPH_TUTOR=true
ENABLE_LANGGRAPH_SPEAKING=true
ENABLE_LANGGRAPH_LISTENING=true
ENABLE_LANGGRAPH_WRITING=true
ENABLE_LANGGRAPH_READING=true
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- MongoDB (local or Atlas)
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env           # then fill in GROQ_API_KEY and MONGODB_URI
```

### Running the Server

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Running Tests

```bash
pytest tests/
```
