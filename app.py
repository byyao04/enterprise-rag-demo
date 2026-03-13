import time
import os
import tempfile
import base64
import hashlib
import secrets
import json
from datetime import datetime
from typing import TypedDict, Annotated
import operator

import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import chromadb
import requests

# LangGraph
from langgraph.graph import StateGraph, END

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
vertexai.init(project=PROJECT_ID, location="us-central1")

st.set_page_config(page_title="Enterprise RAG Demo", page_icon="🤖", layout="wide")

REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://localhost:8501")
SCOPES = "openid email profile"
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ── Agent State ──
class AgentState(TypedDict):
    question: str
    route: str           # "rag" or "web"
    rag_context: str
    web_results: str
    answer: str
    model_name: str


# ── LangGraph Nodes ──
def router_node(state: AgentState) -> AgentState:
    """Decide whether to use RAG or Web Search."""
    llm = ChatVertexAI(model_name=state["model_name"], project=PROJECT_ID, location="us-central1")
    prompt = f"""You are a routing agent. Given a user question, decide if it should be answered using:
- "rag": questions about uploaded documents, internal knowledge, specific files
- "web": questions about current events, recent news, real-time information, or anything not in documents

Question: {state["question"]}

Reply with ONLY one word: "rag" or "web"."""
    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()
    if route not in ["rag", "web"]:
        route = "rag"
    return {**state, "route": route}


def rag_node(state: AgentState) -> AgentState:
    """Retrieve context from vectorstore."""
    retriever = st.session_state.get(f"retriever_{st.session_state.get('user_hash', '')}")
    if retriever:
        docs = retriever.invoke(state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        context = "No documents uploaded."
    return {**state, "rag_context": context}


def web_search_node(state: AgentState) -> AgentState:
    """Search the web using Tavily."""
    tool = TavilySearchResults(max_results=3)
    results = tool.invoke(state["question"])
    web_text = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])
    return {**state, "web_results": web_text}


def answer_node(state: AgentState) -> AgentState:
    """Prepare final prompt — streaming happens in UI layer."""
    if state["route"] == "web":
        final_prompt = f"""Answer the question based on the web search results below.
Web Results:
{state["web_results"]}

Question: {state["question"]}
Answer:"""
    else:
        final_prompt = f"""Answer the question based only on the context below.
Context:
{state["rag_context"]}

Question: {state["question"]}
Answer:"""
    return {**state, "answer": final_prompt}


def route_decision(state: AgentState) -> str:
    return state["route"]


@st.cache_resource
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "rag": "rag",
        "web": "web_search",
    })
    graph.add_edge("rag", "answer")
    graph.add_edge("web_search", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


# ── OAuth helpers ──
@st.cache_resource
def load_client_config():
    secret_name = os.environ.get("SECRET_NAME")
    if secret_name:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(name=secret_name)
        config = json.loads(response.payload.data.decode("utf-8"))
    else:
        with open("/Users/boyuan/gcp-demo/client_secret.json") as f:
            config = json.load(f)
    return config


def generate_pkce():
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return code_verifier, code_challenge


def get_auth_url(client_id, code_challenge, state):
    import urllib.parse
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_code(client_id, client_secret, code, code_verifier):
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    })
    return resp.json()


def get_user_info(access_token):
    resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    return resp.json()


def get_user_hash(email):
    return hashlib.md5(email.encode()).hexdigest()[:8]


CHROMA_BASE_DIR = os.path.expanduser("~/.rag_demo_chroma")

def rebuild_vectorstore(user_hash, docs_dict, embeddings):
    if os.environ.get("CLOUD_RUN"):
        chroma_client = chromadb.EphemeralClient()
    else:
        persist_dir = os.path.join(CHROMA_BASE_DIR, user_hash)
        os.makedirs(persist_dir, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection_name = f"user_{user_hash}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    all_texts, all_metadatas = [], []
    for doc_name, doc_info in docs_dict.items():
        for item in doc_info["splits"]:
            all_texts.append(item["text"])
            all_metadatas.append({"source": doc_name, **item["metadata"]})
    if all_texts:
        vectorstore.add_texts(all_texts, metadatas=all_metadatas)
    return vectorstore


# ── Load config ──
@st.cache_resource
def get_client_credentials():
    config = load_client_config()
    web = config["web"]
    return web["client_id"], web["client_secret"]

CLIENT_ID, CLIENT_SECRET = get_client_credentials()

# ── Handle OAuth callback ──
params = st.query_params
if "code" in params and not st.session_state.get("logged_in"):
    code = params["code"]
    state_param = params.get("state", "")
    try:
        code_verifier = base64.urlsafe_b64decode(state_param + "==").decode()
    except Exception:
        code_verifier = None
    if code_verifier:
        token_data = exchange_code(CLIENT_ID, CLIENT_SECRET, code, code_verifier)
        if "access_token" in token_data:
            user_info = get_user_info(token_data["access_token"])
            email = user_info.get("email", "")
            ALLOWED_EMAILS = os.environ.get("ALLOWED_EMAILS", "")
            if ALLOWED_EMAILS:
                allowed = [e.strip() for e in ALLOWED_EMAILS.split(",")]
                if email not in allowed:
                    st.query_params.clear()
                    st.error(f"⛔ Access denied: {email} is not authorized.")
                    st.stop()
            st.session_state["user"] = {"name": user_info.get("name", "User"), "email": email}
            st.session_state["logged_in"] = True
            st.query_params.clear()
            st.rerun()
        else:
            st.error(f"Token exchange failed: {token_data}")
            st.query_params.clear()
    else:
        st.error("Could not recover session. Please try again.")
        st.query_params.clear()

# ── Auth Gate ──
if not st.session_state.get("logged_in"):
    st.title("🤖 Enterprise RAG Demo")
    st.markdown("### Please sign in to continue")
    code_verifier, code_challenge = generate_pkce()
    state = base64.urlsafe_b64encode(code_verifier.encode()).rstrip(b"=").decode()
    auth_url = get_auth_url(CLIENT_ID, code_challenge, state)
    st.link_button("🔐 Sign in with Google", auth_url)
    st.stop()

# ── Logged in ──
user = st.session_state.get("user", {})
user_email = user.get("email", "anonymous")
user_hash = get_user_hash(user_email)
st.session_state["user_hash"] = user_hash

k_docs = f"docs_{user_hash}"
k_retriever = f"retriever_{user_hash}"
k_messages = f"messages_{user_hash}"
k_active = f"active_doc_{user_hash}"

if k_docs not in st.session_state:
    st.session_state[k_docs] = {}
if k_messages not in st.session_state:
    st.session_state[k_messages] = []

# ── Sidebar ──
with st.sidebar:
    st.markdown(f"👤 **{user.get('name')}**")
    st.caption(user_email)
    st.divider()

    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    chunk_size = st.slider("Chunk Size", 200, 2000, 1500, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100, 10)
    st.divider()

    st.header("📁 My Documents")
    docs = st.session_state[k_docs]
    if not docs:
        st.caption("No documents uploaded yet.")
    else:
        doc_options = ["🗂 All documents"] + list(docs.keys())
        active_label = st.session_state.get(k_active) or "🗂 All documents"
        if active_label not in doc_options:
            active_label = "🗂 All documents"
        selected = st.selectbox("Query scope", doc_options, index=doc_options.index(active_label))
        st.session_state[k_active] = None if selected == "🗂 All documents" else selected
        st.divider()
        st.caption(f"{len(docs)} document(s) in knowledge base")
        for doc_name, doc_info in list(docs.items()):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"📄 **{doc_name[:20]}{'...' if len(doc_name)>20 else ''}**")
            col1.caption(doc_info["uploaded_at"])
            if col2.button("🗑", key=f"del_{doc_name}"):
                del st.session_state[k_docs][doc_name]
                if st.session_state[k_docs]:
                    embeddings = VertexAIEmbeddings(model_name="text-embedding-005", project=PROJECT_ID, location="us-central1")
                    vs = rebuild_vectorstore(user_hash, st.session_state[k_docs], embeddings)
                    st.session_state[k_retriever] = vs.as_retriever(search_kwargs={"k": 3})
                else:
                    if k_retriever in st.session_state:
                        del st.session_state[k_retriever]
                st.rerun()

    st.divider()
    if st.button("🗑 Clear chat history"):
        st.session_state[k_messages] = []
        st.rerun()
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# ── Main ──
st.title("🤖 Enterprise RAG Demo")
st.caption("Powered by Vertex AI + LangGraph Multi-Agent")

# Upload
with st.expander("📤 Upload Documents", expanded=not bool(st.session_state[k_docs])):
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("➕ Add to Knowledge Base"):
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005", project=PROJECT_ID, location="us-central1")
        new_count = 0
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state[k_docs]:
                st.warning(f"'{uploaded_file.name}' already exists, skipping.")
                continue
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs_loaded = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                splits = splitter.split_documents(docs_loaded)
                st.session_state[k_docs][uploaded_file.name] = {
                    "splits": [{"text": s.page_content, "metadata": s.metadata} for s in splits],
                    "uploaded_at": datetime.now().strftime("%m/%d %H:%M"),
                    "chunks": len(splits),
                }
                os.unlink(tmp_path)
                new_count += 1
        if new_count > 0:
            with st.spinner("Rebuilding knowledge base..."):
                vs = rebuild_vectorstore(user_hash, st.session_state[k_docs], embeddings)
                st.session_state[k_retriever] = vs.as_retriever(search_kwargs={"k": 3})
            total_chunks = sum(d["chunks"] for d in st.session_state[k_docs].values())
            st.success(f"✅ Added {new_count} file(s)! {len(st.session_state[k_docs])} docs, {total_chunks} chunks.")
            st.rerun()

# Status bar
if st.session_state[k_docs]:
    active_doc = st.session_state.get(k_active)
    if active_doc:
        st.info(f"🎯 Querying: **{active_doc}** only")
    else:
        total_chunks = sum(d["chunks"] for d in st.session_state[k_docs].values())
        st.info(f"🗂 Querying all **{len(st.session_state[k_docs])} documents** ({total_chunks} chunks)")

# ── Chat ──
st.header("💬 Ask Questions")
st.caption("🤖 Router Agent will automatically decide: search your documents or the web")

for msg in st.session_state[k_messages]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "route" in msg:
            badge = "📄 RAG" if msg["route"] == "rag" else "🌐 Web Search"
            st.caption(f"Agent used: {badge}")

if prompt := st.chat_input("Ask anything — documents or current events..."):
    st.session_state[k_messages].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Router & retrieval..."):
            router_start = time.time()
            graph = build_graph()
            initial_state = AgentState(
                question=prompt,
                route="",
                rag_context="",
                web_results="",
                answer="",
                model_name=model_choice,
            )
            result = graph.invoke(initial_state)
            router_time = time.time() - router_start

        route_used = result["route"]
        final_prompt = result["answer"]
        badge = "📄 RAG" if route_used == "rag" else "🌐 Web Search"
        st.caption(f"🤖 Agent: {badge}  |  Retrieval: {router_time:.2f}s")

        # Stream final answer + measure TTFT and TPOT
        model = GenerativeModel(model_choice)
        stream_start = time.time()
        first_token_time = None
        full_response = ""
        token_count = 0
        placeholder = st.empty()

        for chunk in model.generate_content(final_prompt, stream=True):
            if chunk.text:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - stream_start
                full_response += chunk.text
                token_count += len(chunk.text.split())
                placeholder.markdown(full_response + "▌")

        stream_end = time.time()
        placeholder.markdown(full_response)

        total_latency = stream_end - stream_start
        generation_time = stream_end - (first_token_time or stream_start)
        tpot = (generation_time / token_count * 1000) if token_count > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agent", badge)
        col2.metric("TTFT", f"{ttft:.2f}s")
        col3.metric("TPOT", f"{tpot:.0f}ms/tok")
        col4.metric("Total", f"{total_latency:.2f}s")

    st.session_state[k_messages].append({
        "role": "assistant",
        "content": full_response,
        "route": route_used,
    })

