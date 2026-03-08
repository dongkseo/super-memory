# mcp-super-memory

[![PyPI version](https://img.shields.io/pypi/v/mcp-super-memory)](https://pypi.org/project/mcp-super-memory/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-super-memory)](https://pypi.org/project/mcp-super-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**N:M associative memory graph for LLM agents — delivered as an MCP server.**

> Search **"Newton"** → reach **"strawberry"** through shared keys.
> Embedding similarity alone can't do this.

`mcp-super-memory` is an associative memory system for LLM agents built on a **Key/Value graph** — not a vector store. Memories live in a **Value Space**, accessed through a separate **Key Space** — one memory reachable via many keys, one key leading to many memories. This enables human-like associative leaps (multi-hop graph traversal) that pure embedding search fundamentally cannot replicate.

**Works with:** Claude Desktop · Claude Code · any MCP-compatible LLM agent

---

## Why Not Just Embeddings?

Every existing memory system (Mem0, A-MEM, MemGPT) stores memories as nodes and retrieves them by embedding similarity. This works until it doesn't:

```
Query: "Newton"
Embedding search finds: "Newton discovered gravity" ✅
Embedding search misses: "user likes strawberries"   ❌
```

Super Memory finds both — because "Newton" → apple memory → fruit key → strawberry memory. The **path exists in the key graph**, not in embedding space.

---

## How It Works

```
Key Space (concepts)         Value Space (memories)
─────────────────────        ──────────────────────────────
[Newton]  ──────────────────→ "Newton discovered gravity"
[apple]   ────────┬─────────→      ↑ same memory
[gravity] ────────┘
                  │
[apple]   ────────┼─────────→ "apples are red fruit"
[fruit]   ──────┬─┘
[red]     ──────┤
                │
[fruit]   ──────┼─────────→ "user likes strawberries"
[strawberry]────┘
```

Search `"Newton"` → matches `[Newton]`, `[apple]` keys (1-hop) → follows shared `[fruit]` key → reaches strawberry memory (2-hop, score decayed by 0.3×).

**Results include `hop` field** — you always know if a result is direct or associative.

---

## Key Features

| Feature | Super Memory | A-MEM | Mem0 | MemGPT |
|---------|-------------|-------|------|--------|
| Key/Value separation | ✅ N:M | ❌ | ❌ | ❌ |
| Associative multi-hop | ✅ built-in | ❌ | ❌ | ❌ |
| Depth system | ✅ | ❌ | ❌ | partial |
| Memory versioning | ✅ supersede | overwrites | overwrites | ❌ |
| Time decay | ✅ depth-weighted | ❌ | ❌ | ❌ |
| Key types | ✅ concept/name/proper_noun | ❌ | ❌ | ❌ |
| Key merge (IDF) | ✅ | ❌ | ❌ | ❌ |
| Dual-path recall | ✅ key + content | ❌ | ❌ | ❌ |

### Depth System

Every memory has a depth score `0.0 → 1.0`:

| Stage | Depth | Behavior |
|-------|-------|----------|
| Shallow | `< 0.3` | Recent, unverified. Easy to update or forget. |
| Medium | `0.3–0.7` | Confirmed multiple times. Stable. |
| Deep | `> 0.7` | Well-established fact. Resists correction. |

Depth increases `+0.05` each recall. Deep memories decay slower over time. If you try to correct a deep memory, it resists — its depth stays higher even after supersede.

### Key Types

Not all keys should behave the same. Names shouldn't match semantically — "동건" shouldn't match "뉴턴" just because they're both short Korean words.

| Type | Matching | Use Case |
|------|----------|----------|
| `concept` (default) | Embedding similarity ≥ 0.35 | Topics, categories, attributes |
| `name` | Exact match only | Person names |
| `proper_noun` | Exact match only | Brands, places |

Name/proper_noun keys also get IDF penalty (`×0.5`) when they become hub keys connected to many memories, preventing them from polluting unrelated searches.

### Versioning (not overwriting)

```
"user lives in Seoul"   (depth: 0.4 → weakened to 0.12, preserved)
        ↑ superseded by
"user moved to Busan"   (depth: 0.0, new)
```

Unlike A-MEM which overwrites memory on evolution, Super Memory keeps the full history. Every correction is traceable — when did the belief change, and from what session?

### Key Merging

```
Add key "파이썬"  → finds existing "Python" (similarity 0.87 > threshold 0.85)
                 → reuses existing key instead of creating duplicate
```

Prevents key space fragmentation. Same concept across languages or phrasing stays unified.

### Dual-Path Recall

Recall searches two paths simultaneously:

- **Path A (key matching):** Query embedding → match keys → follow links → memories
- **Path B (content matching):** Query embedding → directly compare against memory content embeddings

Scores from both paths are summed. This ensures memories are found even when they weren't tagged with the right keys.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Key Space                          │
│   [name] [동건] [programming] [python] [fruit] [red]   │
│      ↓      ↓         ↓           ↓       ↓      ↓     │
│   [vec]  [exact]    [vec]       [vec]   [vec]  [vec]   │
└────────────────────────┬────────────────────────────────┘
                         │ N:M links
                         ↓
┌─────────────────────────────────────────────────────────┐
│                     Value Space                         │
│   "user's name is Donggeon"     depth: 0.85  (deep)    │
│   "user likes Python"           depth: 0.30  (medium)  │
│   "user likes strawberries"     depth: 0.05  (shallow) │
└─────────────────────────────────────────────────────────┘
```

**Recall algorithm (2-hop):**
1. Embed query → find matching keys (concept: similarity ≥ 0.35, name/proper_noun: exact match)
2. Also compare query embedding directly against memory content embeddings (≥ 0.3)
3. Follow links → collect memories, aggregate scores (multiple key matches sum up, IDF-weighted)
4. For each 1-hop memory: follow *its* keys → find 2-hop memories (score × `HOP_DECAY = 0.3`)
5. Apply depth factor (`0.5 + depth × 0.5`) and time decay (depth-weighted, 30-day half-life)
6. Return ranked results with `hop` field

---

## MCP Tools

The memory system exposes 8 tools via MCP:

| Tool | Description |
|------|-------------|
| `recall(query, top_k)` | N:M search with 2-hop associative traversal + content matching |
| `remember(content, keys, key_types?)` | Save memory with key concepts and optional type annotations |
| `correct(memory_id, content, keys?)` | Versioned update — old memory preserved but weakened |
| `related(memory_id)` | Find memories sharing keys (associative exploration) |
| `forget(memory_id)` | Permanently delete |
| `get_conversation(session_id, turn?)` | Load original conversation turns |
| `list_memories()` | List all stored memories with keys, depth, access count |
| `memory_stats()` | Get current key/memory/link counts |

A system prompt template is also available via `memory_system_prompt` MCP prompt — include it to instruct the agent to recall silently, use diverse keys, and never mention the memory system to users.

---

## Quick Start (MCP Server)

### Claude Desktop

Add to `claude_desktop_config.json`:

**OpenAI embeddings:**
```json
{
  "mcpServers": {
    "mcp-super-memory": {
      "command": "uvx",
      "args": ["mcp-super-memory"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

**Local embeddings (no API key required):**
```json
{
  "mcpServers": {
    "mcp-super-memory": {
      "command": "uvx",
      "args": ["mcp-super-memory[local]"],
      "env": {
        "EMBEDDING_BACKEND": "local"
      }
    }
  }
}
```

### Claude Code

```bash
# OpenAI embeddings
claude mcp add mcp-super-memory -e OPENAI_API_KEY=your-openai-api-key -- uvx mcp-super-memory

# Local embeddings (no API key required)
claude mcp add mcp-super-memory -e EMBEDDING_BACKEND=local -- uvx "mcp-super-memory[local]"
```

### Manual / Development

```bash
git clone https://github.com/donggyun112/mcp-super-memory
cd super-memory
```

Create `.env`:
```
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Or use local embeddings (no API key required):
```
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2  # optional, this is the default
```

> **Note:** Mixing backends on existing data will break recall. If switching backends, clear `~/.super-memory/graph.json` first.

```bash
uv sync
uv run mcp-super-memory
```

**Requirements:**
- Python 3.12+
- OpenAI API key (for embeddings) — or `sentence-transformers` for local embeddings

---

## Data Storage

All data is local. No external database required.

```
data/
├── graph.json          # keys, memories, links
└── conversations/
    └── {session_id}.jsonl   # original conversation turns
```

---

## Limitations

- **Linear scan** — suitable for personal use (~10k memories). FAISS/ChromaDB integration planned for larger scale.
- **2-hop max** — deeper associative chains require `related()` tool calls by the agent.
- **Agent quality matters** — key selection on `remember` affects retrieval quality. System prompt tuning is important.

---

## Comparison with A-MEM

A-MEM (NeurIPS 2025) focuses on *memory evolution* — when new memories arrive, existing memories' descriptions update. Super Memory focuses on *memory access* — how to reach the right memory through associative paths.

They solve different problems. A-MEM asks "how do we keep memories well-organized?" Super Memory asks "how do we find memories the way humans actually think?"

The versioning approach also differs: A-MEM overwrites on evolution (current state only), Super Memory preserves history (full timeline).

---

## Roadmap

- [ ] FAISS/ChromaDB for scale
- [ ] Coding agent profile (different key strategies for code context)
- [ ] Memory export/import
- [ ] Multi-user support

---

## License

MIT
