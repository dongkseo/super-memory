import json

from mcp.server.fastmcp import FastMCP

try:
    from .memory_graph import MemoryGraph, save_turn, load_conversation
except ImportError:
    from memory_graph import MemoryGraph, save_turn, load_conversation

mcp = FastMCP(
    "super-memory",
    instructions=(
        "You have long-term memory. Use it silently — never tell the user you're saving or recalling.\n\n"
        "## When to recall\n"
        "- ALWAYS at the start of a new conversation (recall the user's name, preferences, context)\n"
        "- Before answering any personal question ('what do I like?', 'who am I?')\n"
        "- When the topic might connect to something previously discussed\n"
        "- Use SPECIFIC queries: recall('이름'), recall('좋아하는 음식') — not recall('사용자 정보')\n\n"
        "## When to remember\n"
        "- User shares personal info (name, preferences, facts about themselves)\n"
        "- User corrects you on something\n"
        "- Important decisions or conclusions from the conversation\n"
        "- Do NOT save trivial or temporary things (e.g. 'user said hello')\n\n"
        "## Key selection (critical for retrieval quality)\n"
        "- Keys are search terms that should lead to this memory. Think: 'what would someone search to find this?'\n"
        "- Use 3-6 diverse keys: mix categories, attributes, and specifics\n"
        "- Example: 'user likes strawberries' → keys: ['fruit', 'strawberry', 'food preference', 'likes']\n"
        "- Set key_types for names: {'동건': 'name', 'Apple': 'proper_noun'}\n\n"
        "## Behavior\n"
        "- Act like you naturally know things. Never say '기억에 의하면' or '메모리에서 찾았어요'.\n"
        "- Use correct() when info changes, not remember(). Use forget() only for truly wrong info."
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
    return f"{len(graph.keys)} keys, {len(graph.memories)} memories, {graph.link_count} links"


@mcp.prompt()
def memory_system_prompt() -> str:
    """System prompt for LLM agents using super-memory. Include this in your system prompt."""
    return MEMORY_SYSTEM.format(stats=_stats())


@mcp.tool()
async def recall(query: str, top_k: int = 5) -> str:
    """Search memory. Call this BEFORE answering personal questions or at conversation start. Use specific queries — recall('이름') is better than recall('사용자 정보'). Returns memories ranked by relevance with hop=1 (direct) or hop=2 (associative). Memories get stronger each time they're recalled."""
    results = await graph.recall(query, top_k)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def remember(content: str, keys: list[str], key_types: dict[str, str] | None = None) -> str:
    """Save important information to memory. Use when user shares personal info, preferences, or facts worth keeping. Keys are search terms — think 'what would I search to find this later?' Use 3-6 diverse keys mixing categories and specifics. Example: content='user likes strawberries', keys=['fruit', 'strawberry', 'food preference', 'likes']. Set key_types={'동건': 'name'} for person names, {'Apple': 'proper_noun'} for brands."""
    if isinstance(keys, str):
        try:
            keys = json.loads(keys)
        except (json.JSONDecodeError, TypeError):
            keys = [keys]
    mid = await graph.add(content, keys, key_types=key_types)
    return json.dumps({"saved": mid})


@mcp.tool()
async def correct(memory_id: str, content: str, keys: list[str] | None = None, key_types: dict[str, str] | None = None) -> str:
    """Update outdated information. Use when user corrects you or info changes (e.g. moved cities, changed job). Old version is preserved but weakened — never lost. Omit keys to keep the same search terms. Deep memories (frequently recalled) resist correction."""
    nid = await graph.supersede(memory_id, content, key_concepts=keys, key_types=key_types)
    return json.dumps({"new_id": nid, "superseded": memory_id})


@mcp.tool()
def related(memory_id: str) -> str:
    """Explore connections from a specific memory. Returns other memories that share keys with it. Use after recall() to dig deeper into a topic or discover unexpected associations."""
    results = graph.get_related(memory_id)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def forget(memory_id: str) -> str:
    """Permanently delete a memory. Only use for completely wrong information. For outdated info, use correct() instead — it preserves history."""
    ok = await graph.delete(memory_id)
    return json.dumps({"deleted": ok})


@mcp.tool()
def get_conversation(session_id: str, turn: int | None = None) -> str:
    """Load raw conversation turns from a past session. Use when a recalled memory lacks detail and you need the original context."""
    turns = load_conversation(session_id, turn)
    return json.dumps(turns, ensure_ascii=False)


@mcp.tool()
def list_memories() -> str:
    """List all stored memories. Use for debugging or when you need to browse everything. Prefer recall() for normal retrieval."""
    results = graph.list_all()
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def memory_stats() -> str:
    """Get counts of keys, memories, and links in the system."""
    return json.dumps({
        "keys": len(graph.keys),
        "memories": len(graph.memories),
        "links": len(graph.links),
    })
