# 🤖 Enterprise RAG Demo

A production-grade Retrieval-Augmented Generation (RAG) system built on Google Cloud Platform, featuring a LangGraph multi-agent architecture, Google OAuth authentication, and MCP (Model Context Protocol) integration for Claude Desktop.

## ✨ Features

- **Multi-Agent RAG Pipeline** — LangGraph Router Agent automatically decides between document search (RAG) or live web search (Tavily)
- **Google OAuth 2.0** — Secure PKCE-based authentication with email whitelist
- **Multi-file PDF Upload** — Upload and manage multiple documents with per-user isolation
- **Streaming Responses** — Real-time answer generation with TTFT / latency metrics
- **MCP Integration** — Query your documents directly from Claude Desktop via Model Context Protocol
- **Persistent Vectorstore** — Chroma persistent storage enables cross-process document access
- **Cloud Run Ready** — Fully containerized, deployable to Google Cloud Run

## 🏗️ Architecture

### Local Development

```
┌─────────────────────────────────────────────────────┐
│                    Claude Desktop                    │
│                  (MCP Client)                        │
└──────────────────────┬──────────────────────────────┘
                       │ MCP Protocol (stdio)
                       ▼
               mcp_server.py
                       │ HTTP POST /query
                       ▼
              api.py (FastAPI :8081)  ◄──── must be running locally
                       │
                       ▼
         Chroma PersistentClient
         (~/.rag_demo_chroma/{user_hash}/)
                       ▲
                       │ indexes on upload
                       │
┌──────────────────────┴──────────────────────────────┐
│             app.py (Streamlit :8501)                 │
│                                                      │
│  User uploads PDF ──► LangGraph Multi-Agent          │
│                         ├── Router Node              │
│                         ├── RAG Node (Chroma)        │
│                         ├── Web Search Node (Tavily) │
│                         └── Answer Node              │
│                       All powered by Gemini 2.5 Flash│
└─────────────────────────────────────────────────────┘
```

### Cloud Run Deployment

```
┌─────────────────────────────────────────────────────┐
│                      Browser                         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS
                       ▼
┌─────────────────────────────────────────────────────┐
│              Google Cloud Run                        │
│                                                      │
│  app.py (Streamlit :8080)                            │
│      │                                               │
│      ├── Google OAuth ──► Secret Manager             │
│      │                                               │
│      └── LangGraph Multi-Agent                       │
│               ├── Router Node                        │
│               ├── RAG Node ──► Chroma EphemeralClient│
│               │                (in-memory only)      │
│               ├── Web Search Node ──► Tavily          │
│               └── Answer Node                        │
│                     │                                │
│                     ▼                                │
│              Vertex AI (Gemini 2.5 Flash)            │
└─────────────────────────────────────────────────────┘

Note: MCP integration is local-only and not available on Cloud Run.
      Chroma EphemeralClient means documents are lost on container restart.
```

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI API enabled
- Google OAuth 2.0 credentials
- Tavily API key ([get one free](https://tavily.com))
- Claude Desktop (for MCP integration)

### 1. Clone & Install

```bash
git clone https://github.com/byyao04/enterprise-rag-demo.git
cd enterprise-rag-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export TAVILY_API_KEY="your-tavily-api-key"
```

Then reload:
```bash
source ~/.zshrc
```

> ⚠️ **Every new terminal session** requires `source ~/.zshrc` (or open a new terminal after saving). Without this, `GCP_PROJECT_ID` will be empty and Vertex AI calls will fail.

### 3. Set Up Application Default Credentials (ADC)

Vertex AI requires Google credentials to be configured locally:

```bash
gcloud auth application-default login
```

This opens a browser window — sign in with the Google account that has access to your GCP project. You only need to do this once per machine.

### 4. Set Up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID (Web Application)
3. Add `http://localhost:8501` as an authorized redirect URI
4. Download the credentials as `client_secret.json` and place it in the project root

> ⚠️ **Never commit `client_secret.json` to git!** It's already in `.gitignore`.

### 5. Run Locally

You need **two terminals** running simultaneously:

**Terminal 1 — Streamlit UI:**
```bash
source venv/bin/activate
streamlit run app.py
```

**Terminal 2 — FastAPI backend (required for MCP):**
```bash
source venv/bin/activate
python api.py
```

Open http://localhost:8501, sign in with Google, and upload a PDF.

> ⚠️ **api.py must be running** whenever you want to use MCP from Claude Desktop. If api.py is not running, Claude Desktop queries will return a 500 error.

### 6. MCP Integration with Claude Desktop

> **Prerequisites**: Both `streamlit run app.py` and `python api.py` must be running, and you must have uploaded at least one document via the Streamlit UI first.

**Step 1** — Get your user hash (derived from your Google login email):
```bash
python3 -c "import hashlib; print(hashlib.md5('your@email.com'.encode()).hexdigest()[:8])"
```

**Step 2** — Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "enterprise-rag": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/enterprise-rag-demo/mcp_server.py"],
      "env": {
        "RAG_USER_HASH": "your-user-hash"
      }
    }
  }
}
```

**Step 3** — Restart Claude Desktop.

**Step 4** — Ask Claude Desktop:
> "Please use query_documents to search for [your question]"

Claude Desktop will query your locally uploaded documents via the MCP server → api.py → Chroma pipeline.

## 💾 Data Storage

### Where are uploaded documents stored?

Uploaded PDFs are **not stored as files**. Instead, they are parsed, split into chunks, and converted into vector embeddings via Vertex AI. Only the embeddings are persisted.

| Environment | Storage | Persistence |
|-------------|---------|-------------|
| Local | `~/.rag_demo_chroma/{user_hash}/` (Chroma PersistentClient) | ✅ Survives restarts |
| Cloud Run | In-memory (Chroma EphemeralClient) | ❌ Lost on container restart |

> **Cloud Run limitation**: Cloud Run containers are stateless and may be recycled when idle. Any documents uploaded in a session will need to be re-uploaded after a container restart.

### Upgrading to persistent Cloud storage

For production use, consider replacing the EphemeralClient with one of these options:

- **GCS + re-index on startup** — Store original PDFs in Google Cloud Storage and re-build the vectorstore when the container starts
- **Vertex AI Vector Search** — Fully managed vector database on GCP
- **Cloud Firestore** — Store embeddings as documents in a NoSQL database

## ☁️ Deploy to Cloud Run

```bash
# Build and push Docker image
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag-demo/rag-demo

# Deploy
gcloud run deploy rag-demo \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag-demo/rag-demo \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars \
    GCP_PROJECT_ID=YOUR_PROJECT_ID,\
    SECRET_NAME=projects/YOUR_PROJECT_NUMBER/secrets/oauth-client-secret/versions/latest,\
    REDIRECT_URI=https://YOUR_CLOUD_RUN_URL,\
    TAVILY_API_KEY=YOUR_TAVILY_KEY,\
    ALLOWED_EMAILS=your@email.com,\
    CLOUD_RUN=1 \
  --memory 2Gi
```

> For Cloud Run, store your OAuth credentials in **Secret Manager** and reference via `SECRET_NAME`.

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GCP_PROJECT_ID` | Your GCP project ID | ✅ |
| `TAVILY_API_KEY` | Tavily search API key | ✅ |
| `SECRET_NAME` | GCP Secret Manager path for OAuth credentials | Cloud Run only |
| `REDIRECT_URI` | OAuth redirect URI | Cloud Run only |
| `ALLOWED_EMAILS` | Comma-separated list of allowed emails | Optional |
| `CLOUD_RUN` | Set to `1` when running on Cloud Run | Cloud Run only |

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI + Uvicorn
- **AI/ML**: Vertex AI (Gemini 2.5 Flash, text-embedding-005)
- **Orchestration**: LangGraph
- **Vector DB**: ChromaDB (persistent)
- **Web Search**: Tavily
- **Auth**: Google OAuth 2.0 PKCE
- **MCP**: Model Context Protocol
- **Cloud**: Google Cloud Run, Secret Manager, Artifact Registry
