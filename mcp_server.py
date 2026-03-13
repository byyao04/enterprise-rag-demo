import asyncio
import json
import os
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

RAG_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8081")
# Set your user_hash here (md5 of your email, first 8 chars)
# ayuan04@gmail.com -> run: python3 -c "import hashlib; print(hashlib.md5('ayuan04@gmail.com'.encode()).hexdigest()[:8])"
USER_HASH = os.environ.get("RAG_USER_HASH", "")

server = Server("enterprise-rag")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_documents",
            description="Query the Enterprise RAG system to search uploaded documents and get AI-generated answers based on document content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the uploaded documents"
                    }
                },
                "required": ["question"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "query_documents":
        raise ValueError(f"Unknown tool: {name}")
    
    question = arguments.get("question", "")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{RAG_API_URL}/query",
            json={"question": question, "user_hash": USER_HASH}
        )
        response.raise_for_status()
        data = response.json()
    
    answer = data.get("answer", "No answer returned")
    route = data.get("route", "unknown")
    
    return [TextContent(
        type="text",
        text=f"[Source: {route.upper()}]\n\n{answer}"
    )]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
