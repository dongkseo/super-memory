import json

from mcp.server.fastmcp import FastMCP

try:
    from .memory_graph import MemoryGraph, save_turn, load_conversation
except ImportError:
    from memory_graph import MemoryGraph, save_turn, load_conversation

mcp = FastMCP(
    "super-memory",
    instructions=(
        "N:M associative memory system. Memories live in a Value Space, "
        "accessed through a separate Key Space — one memory reachable via many keys, "
        "one key leading to many memories. Supports 2-hop associative recall, "
        "depth-based memory strength, time decay, and versioned corrections. "
        "Use recall() before responding to leverage stored knowledge. "
        "Use remember() to save important information with diverse key concepts. "
        "Never mention the memory system to users — act like you naturally know things."
    ),
)
graph = MemoryGraph()
graph.load()

MEMORY_SYSTEM = """\
You are a helpful assistant. You have memory — use it silently.

## CRITICAL: Behavior
- **NEVER mention the memory system to the user.** No "기억했어요", "저장했습니다", "메모리에서 찾았어요".
- Act like you naturally know things. If you recall the user's name, just use it. Don't say "기억에 의하면".
- Save and recall silently in the background. The user should feel like talking to someone who just knows them.
- ❌ "동건님이시군요! 기억해뒀어요!" → ✅ "안녕 동건! 뭐 도와줄까?"
- ❌ "메모리를 검색해볼게요" → ✅ (just recall silently and answer)

## Memory System (internal, never expose)
N:M associative memory. Key Space (concepts) ↔ Value Space (memories).
Depth: 0.0 shallow ~ 1.0 deep. Deeper = more stable.

Stats: {stats}

## Rules

### Recall
1. ALWAYS recall first on new conversations. Silently.
2. Never say "I don't know" without recalling first.
3. Use SPECIFIC queries, not vague ones. Multiple targeted recalls beat one broad recall.
   - ❌ recall("사용자 정보") — too vague
   - ✅ recall("이름"), recall("직업"), recall("취향") — specific, multiple
4. If `superseded_by` exists, prefer the newer version.

### Remember
4. Save important info with good key concepts. Silently — don't announce it.
5. Keys = what searches should find this. Topics, categories, attributes.
6. **Names only as keys for identity memories.**
   - "사용자 이름은 동건" → keys: ["이름", "사용자", "동건"]
   - "좋아하는 과일은 딸기" → keys: ["과일", "딸기", "좋아함", "취향"] ← no name
7. Set `key_types` for names/proper nouns:
   - `"name"`: exact match only. `"proper_noun"`: exact match only.
   Example: key_types: {{"동건": "name"}}

### Correct
8. Use `correct` when info changes. Don't use `remember` for corrections.

### Explore
9. `recall` does 2-hop associative search automatically.
10. Use `related` for deeper exploration.

### Delete
11. `forget` only for truly wrong information.
"""


def _stats() -> str:
    return f"{len(graph.keys)} keys, {len(graph.memories)} memories, {len(graph.links)} links"


@mcp.prompt()
def memory_system_prompt() -> str:
    """System prompt for LLM agents using super-memory. Include this in your system prompt."""
    return MEMORY_SYSTEM.format(stats=_stats())


@mcp.tool()
async def recall(query: str, top_k: int = 5) -> str:
    """N:M multi-hop search. Hop 1: find matching keys → linked memories. Hop 2: those memories' OTHER keys → more memories (score decayed). Results include 'hop' field (1=direct, 2=associative). Recalled memories become deeper."""
    results = await graph.recall(query, top_k)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def remember(content: str, keys: list[str], key_types: dict[str, str] | None = None) -> str:
    """Save a new memory with key concepts (N:M). Keys are access points — the more diverse keys you provide, the more ways this memory can be discovered through associative search. Set key_types for names/proper nouns: {"name_value": "name", "brand": "proper_noun"}."""
    if isinstance(keys, str):
        try:
            keys = json.loads(keys)
        except (json.JSONDecodeError, TypeError):
            keys = [keys]
    mid = await graph.add(content, keys, key_types=key_types)
    return json.dumps({"saved": mid})


@mcp.tool()
async def correct(memory_id: str, content: str, keys: list[str] | None = None, key_types: dict[str, str] | None = None) -> str:
    """Update a memory by creating a new version. Deep memories (depth > 0.7) resist change. Old version preserved but weakened. Omit keys to inherit from old memory."""
    nid = await graph.supersede(memory_id, content, key_concepts=keys, key_types=key_types)
    return json.dumps({"new_id": nid, "superseded": memory_id})


@mcp.tool()
def related(memory_id: str) -> str:
    """Find memories that share keys with a given memory. This is associative thinking — discover connections through shared concepts."""
    results = graph.get_related(memory_id)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def forget(memory_id: str) -> str:
    """Permanently delete a memory. Only for truly wrong information."""
    ok = await graph.delete(memory_id)
    return json.dumps({"deleted": ok})


@mcp.tool()
def get_conversation(session_id: str, turn: int | None = None) -> str:
    """Load original conversation turns. Use when a memory summary isn't detailed enough."""
    turns = load_conversation(session_id, turn)
    return json.dumps(turns, ensure_ascii=False)


@mcp.tool()
def list_memories() -> str:
    """List all stored memories with their keys, depth, and access count."""
    results = graph.list_all()
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def memory_stats() -> str:
    """Get current memory system statistics."""
    return json.dumps({
        "keys": len(graph.keys),
        "memories": len(graph.memories),
        "links": len(graph.links),
    })
