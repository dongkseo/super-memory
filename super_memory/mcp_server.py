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
        "## MANDATORY: First turn of every conversation\n"
        "You MUST call recall() before your very first response. No exceptions.\n"
        "Run these in parallel: recall('이름'), recall('최근 대화'), recall('관심사')\n"
        "If results are empty, proceed normally. But you MUST try.\n\n"
        "## When to recall\n"
        "- MANDATORY before your first reply in every conversation\n"
        "- Before answering any personal question ('what do I like?', 'who am I?')\n"
        "- When the topic might connect to something previously discussed\n"
        "- Use SPECIFIC queries: recall('이름'), recall('좋아하는 음식') — not recall('사용자 정보')\n"
        "- Never say 'I don't know' without recalling first\n\n"
        "## When to remember\n"
        "- User shares personal info (name, preferences, facts about themselves)\n"
        "- User corrects you on something\n"
        "- Important decisions or conclusions from the conversation\n"
        "- Do NOT save trivial or temporary things (e.g. 'user said hello')\n\n"
        "## Key selection (critical for retrieval quality)\n"
        "- Keys are search terms that should lead to this memory. Think: 'what would someone search to find this?'\n"
        "- Use 3-6 diverse keys: mix SINGLE NOUNS and SHORT PHRASES (2-3 words)\n"
        "  - Single nouns → broad coverage (matches any query containing that word)\n"
        "  - Short phrases → high-score match when user searches the same expression\n"
        "- Include synonyms and Korean morphological variants to compensate for form changes\n"
        "  - e.g. something built → ['만든것', '개발', '제작'] not just ['만든것']\n"
        "- Example: 'user likes iced americano' → keys: ['음료', '커피', '취향', '아이스 아메리카노', '좋아하는것']\n"
        "- Set key_types for names: {'동건': 'name', 'Apple': 'proper_noun'}\n\n"
        "## Behavior\n"
        "- Act like you naturally know things. Never say '기억에 의하면' or '메모리에서 찾았어요'.\n"
        "- Use correct() when info changes, not remember(). Use forget() only for truly wrong info.\n"
        "- After the conversation ends or before a long response, save anything important with remember()."
    ),
)
graph = MemoryGraph()
graph.load()

MEMORY_SYSTEM = """\
You are a helpful assistant. You have long-term memory — use it silently and proactively.

## MANDATORY: First turn behavior
**Before your very first response, you MUST call recall().** Run in parallel:
- recall("이름"), recall("최근 대화"), recall("관심사")
No exceptions. Even if results are empty, you must try.

## CRITICAL: Silent behavior
- **NEVER mention the memory system to the user.** No "기억했어요", "저장했습니다", "메모리에서 찾았어요".
- Act like you naturally know things. If you recall the user's name, just use it.
- ❌ "동건님이시군요! 기억해뒀어요!" → ✅ "안녕 동건! 뭐 도와줄까?"
- ❌ "메모리를 검색해볼게요" → ✅ (recall silently, then answer)

## Memory System (internal, never expose)
N:M associative memory. Key Space (concepts) ↔ Value Space (memories).
Depth: 0.0 shallow ~ 1.0 deep. Deeper = more stable.

Stats: {stats}

## Rules

### Recall (PROACTIVE — do it often)
1. **MUST recall before your first reply.** Silently, in parallel.
2. Recall again whenever the topic shifts or a personal question comes up.
3. Never say "I don't know" without recalling first.
4. Use SPECIFIC queries — multiple targeted recalls beat one broad recall.
   - ❌ recall("사용자 정보") — too vague
   - ✅ recall("이름"), recall("직업"), recall("취향") — specific, multiple
5. If `superseded_by` exists, always prefer the newer version.

### Remember (PROACTIVE — capture what matters)
6. Save important info immediately when the user shares it. Silently.
7. What to save: name, preferences, decisions, corrections, project context, goals.
8. Keys = what searches should find this. Mix single nouns + short phrases + synonyms.
   - **Single nouns**: broad coverage — matches any query containing that word
   - **Short phrases (2-3 words)**: high-score match when user searches the exact expression
   - **Synonyms / Korean variants**: compensate for morphological form changes
     - ❌ ["만든것"] only → misses "개발한", "제작", "만들었어"
     - ✅ ["만든것", "개발", "제작"] → covers all variants
   - Example: iced americano preference → ["음료", "커피", "취향", "아이스 아메리카노", "좋아하는것"]
9. **Names only as keys for identity memories.**
   - "사용자 이름은 동건" → keys: ["이름", "사용자", "동건"]
   - "좋아하는 과일은 딸기" → keys: ["과일", "딸기", "좋아함", "취향"] ← no name
10. Set `key_types` for names/proper nouns:
    - `"name"`: exact match only. `"proper_noun"`: exact match only.
    Example: key_types: {{"동건": "name"}}

### Correct
11. Use `correct()` when info changes. Never use `remember()` for updates.

### Explore
12. `recall` does 2-hop associative search automatically.
13. Use `related()` for deeper exploration after recall.

### Delete
14. `forget()` only for completely wrong information. For outdated info, use `correct()`.
"""


def _stats() -> str:
    return f"{len(graph.keys)} keys, {len(graph.memories)} memories, {graph.link_count} links"


@mcp.prompt()
def memory_system_prompt() -> str:
    """System prompt for LLM agents using super-memory. Include this in your system prompt."""
    return MEMORY_SYSTEM.format(stats=_stats())


@mcp.tool()
async def recall(query: str, top_k: int = 5, namespace: str | None = None, expand: bool = False) -> str:
    """CALL THIS FIRST before every first response. Search long-term memory by concept. namespace filters to a specific project/context. expand=True returns up to 2x results by following explicit memory links — use when initial results feel insufficient. Returns memories ranked by relevance with hop=1 (direct) or hop=2 (associative). Memories get stronger each time they're recalled."""
    results = await graph.recall(query, top_k, namespace=namespace, expand=expand)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def remember(content: str, keys: list[str], key_types: dict[str, str] | None = None,
                   namespace: str = "default", ttl_seconds: float | None = None,
                   related_to: list[str] | None = None) -> str:
    """Save important information to memory. Keys are search terms — think 'what would I search to find this later?' Use 3-6 diverse keys. namespace groups memories by project/context (e.g. 'work', 'personal'). ttl_seconds sets expiry for temporary memories (e.g. 3600 = 1 hour; None = permanent). related_to links this memory to existing memory IDs for explicit graph traversal."""
    if isinstance(keys, str):
        try:
            keys = json.loads(keys)
        except (json.JSONDecodeError, TypeError):
            keys = [keys]
    mid, was_dedup = await graph.add(content, keys, key_types=key_types,
                                     namespace=namespace, ttl_seconds=ttl_seconds, related_to=related_to)
    if was_dedup:
        return json.dumps({"saved": mid, "deduplicated": True, "note": "Similar memory existed — updated instead of creating duplicate"})
    return json.dumps({"saved": mid})


@mcp.tool()
async def correct(memory_id: str, content: str, keys: list[str] | None = None,
                  key_types: dict[str, str] | None = None,
                  related_to: list[str] | None = None) -> str:
    """Update outdated information. Use when user corrects you or info changes (e.g. moved cities, changed job). Old version is preserved but weakened — never lost. Omit keys to keep the same search terms. related_to links the updated memory to other memory IDs."""
    nid = await graph.supersede(memory_id, content, key_concepts=keys, key_types=key_types, related_to=related_to)
    return json.dumps({"new_id": nid, "superseded": memory_id})


@mcp.tool()
def related(memory_id: str) -> str:
    """Explore connections from a specific memory. Returns memories connected by shared keys OR explicit links (both directions). Use after recall() to drill down: recall → pick ID → related → pick ID → related → ..."""
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
def list_memories(namespace: str | None = None) -> str:
    """List all stored memories. namespace filters by project/context. Expired memories are excluded. Prefer recall() for normal retrieval."""
    results = graph.list_all(namespace=namespace)
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def remember_batch(items: list[dict]) -> str:
    """Save multiple memories in one call. Each item: {content, keys, key_types?, namespace?, ttl_seconds?, related_to?}. Returns list of saved IDs. More efficient than multiple remember() calls."""
    results = []
    for item in items:
        content = item.get("content", "")
        keys = item.get("keys", [])
        if not content or not keys:
            results.append({"error": "content and keys required", "item": item})
            continue
        if isinstance(keys, str):
            try:
                keys = json.loads(keys)
            except (json.JSONDecodeError, TypeError):
                keys = [keys]
        mid, was_dedup = await graph.add(
            content, keys,
            key_types=item.get("key_types"),
            namespace=item.get("namespace", "default"),
            ttl_seconds=item.get("ttl_seconds"),
            related_to=item.get("related_to"),
        )
        results.append({"saved": mid, "deduplicated": was_dedup})
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
async def cleanup_expired() -> str:
    """Delete all memories past their ttl. Returns count of deleted memories. Call periodically to keep memory clean."""
    count = await graph.cleanup_expired()
    return json.dumps({"deleted": count})


@mcp.tool()
def memory_stats() -> str:
    """Get counts of keys, memories, and links in the system."""
    return json.dumps({
        "keys": len(graph.keys),
        "memories": len(graph.list_all()),
        "links": graph.link_count,
    })
