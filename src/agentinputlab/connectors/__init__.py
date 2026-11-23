"""External tool and knowledge connector contracts."""

from agentinputlab.backends.retrieval import (
    Document,
    InMemoryBM25Retriever,
    RetrievalResult,
    Retriever,
)

from .tools import (
    InMemoryToolConnector,
    MCPToolConnector,
    MCPTransport,
    OpenAPIToolConnector,
    OpenAPITransport,
    SchemaOnlyToolConnector,
    ToolCall,
    ToolConnector,
    ToolResult,
    ToolSpec,
    normalize_json_schema,
    normalize_mcp_tool,
    python_tool_spec,
    validate_json_value,
)

__all__ = [
    "Document",
    "InMemoryBM25Retriever",
    "InMemoryToolConnector",
    "MCPToolConnector",
    "MCPTransport",
    "OpenAPIToolConnector",
    "OpenAPITransport",
    "RetrievalResult",
    "Retriever",
    "SchemaOnlyToolConnector",
    "ToolCall",
    "ToolConnector",
    "ToolResult",
    "ToolSpec",
    "normalize_json_schema",
    "normalize_mcp_tool",
    "python_tool_spec",
    "validate_json_value",
]
