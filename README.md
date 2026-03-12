# Enterprise RAG Demo

A production-grade multi-agent RAG application built on Google Cloud Platform, featuring automated query routing, real-time web search, and per-user document management.

**Live Demo:** https://rag-demo-231380169643.us-central1.run.app *(Access restricted — contact author for demo access)*

---

## Architecture

```
User Question
      ↓
Router Agent (Gemini 2.5 Flash)
      ↓               ↓
RAG Agent         Web Search Agent
(Vertex AI        (Tavily API)
 Embeddings +
 ChromaDB)
      ↓               ↓
       Final Answer (Streaming)
```

## Features

- **Multi-Agent Routing** — LangGraph-powered Router Agent automatically decides whether to retrieve from uploaded documents or search the web in real-time
- **RAG Pipeline** — PDF upload, chunking, Vertex AI text-embedding-005, ChromaDB vector store
- **Web Search** — Tavily API integration for real-time information retrieval
- **Streaming Responses** — Token-by-token streaming with TTFT / TPOT / Total Latency metrics
- **Document Management** — Upload multiple PDFs, delete individual files, switch query scope between documents
- **Google OAuth 2.0** — PKCE flow for secure authentication
- **Per-user Isolation** — Each user's documents and chat history are isolated
- **Access Control** — Email whitelist via environment variable

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Vertex AI Gemini 2.5 Flash |
| Embeddings | Vertex AI text-embedding-005 |
| Agent Framework | LangGraph |
| Vector Store | ChromaDB (in-memory) |
| Web Search | Tavily API |
| Auth | Google OAuth 2.0 + PKCE |
| Frontend | Streamlit |
| Container | Docker |
| Hosting | GCP Cloud Run |
| Secrets | GCP Secret Manager |
| CI/CD | GCP Cloud Build |

---

## Local Development

### Prerequisites
- Python 3.11+
- GCP account with Vertex AI enabled
- Tavily API key — get one free at [app.tavily.com](https://app.tavily.com)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/enterprise-rag-demo.git
cd enterprise-rag-demo

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up Google OAuth credentials

1. Go to [GCP Console](https://console.cloud.google.com) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID (Web application)
3. Add `http://localhost:8501` to Authorized JavaScript origins and Authorized redirect URIs
4. Download the JSON file and save it as `client_secret.json` in the project root

> ⚠️ `client_secret.json` is in `.gitignore` and should never be committed to git.

### 3. Set environment variables

```bash
export TAVILY_API_KEY=your_tavily_key
```

Or add to `~/.zshrc` to persist across terminal sessions:
```bash
echo 'export TAVILY_API_KEY="your_tavily_key"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Deploy to GCP Cloud Run

### 1. Find your Project ID and Project Number

Go to [GCP Console](https://console.cloud.google.com) → click the project dropdown (top left) → your **Project ID** and **Project number** are listed there.

### 2. Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### 3. Store OAuth credentials in Secret Manager

```bash
gcloud secrets create oauth-client-secret \
  --data-file=client_secret.json
```

### 4. Grant permissions to Cloud Run service account

```bash
# Secret Manager access
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Vertex AI access
gcloud projects add-iam-policy-binding <PROJECT_ID> \
  --member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### 5. Create Artifact Registry repository

```bash
gcloud artifacts repositories create rag-demo \
  --repository-format=docker \
  --location=us-central1
```

### 6. Build and deploy

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/<PROJECT_ID>/rag-demo/rag-demo

gcloud run deploy rag-demo \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/rag-demo/rag-demo \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars SECRET_NAME=projects/<PROJECT_NUMBER>/secrets/oauth-client-secret/versions/latest,\
REDIRECT_URI=https://<YOUR_CLOUD_RUN_URL>,\
TAVILY_API_KEY=<YOUR_TAVILY_KEY> \
  --memory 2Gi
```

### 7. Update OAuth redirect URI

After deploy, add your Cloud Run URL to the OAuth client's Authorized redirect URIs and Authorized JavaScript origins in GCP Console.

---

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `TAVILY_API_KEY` | Tavily search API key | Yes |
| `SECRET_NAME` | GCP Secret Manager path for OAuth credentials | Cloud Run only |
| `REDIRECT_URI` | OAuth redirect URI | Cloud Run only |
| `ALLOWED_EMAILS` | Comma-separated whitelist of allowed emails | Optional |

## Access Control (Optional)

To restrict access to specific users:

```bash
gcloud run services update rag-demo \
  --region us-central1 \
  --update-env-vars ALLOWED_EMAILS="user1@gmail.com,user2@gmail.com"
```
