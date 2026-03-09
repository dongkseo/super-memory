import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { MemoryGraph, loadConversation, sanitizeKeys } from "./memoryGraph.js";

function parseArray(v: unknown): unknown[] | null {
  if (Array.isArray(v)) return v;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return Array.isArray(p) ? p : null; } catch { return null; }
  }
  return null;
}

function parseObject(v: unknown): Record<string, unknown> | null {
  if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
  if (typeof v === "string") {
    try { const p = JSON.parse(v); return p && typeof p === "object" ? p : null; } catch { return null; }
  }
  return null;
}

const MEMORY_SYSTEM = `\
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
5. If \`superseded_by\` exists, always prefer the newer version.

### Remember (PROACTIVE — capture what matters)
6. Save important info immediately when the user shares it. Silently.
7. What to save: name, preferences, decisions, corrections, project context, goals.
8. Keys = what searches should find this. **Think like a search engine — include every form someone might use to ask about this.**
   - **Topic noun**: what category is this? (거주지, 음료, 반려동물, 언어)
   - **Specific noun**: the actual value (성수동, 아메리카노, 고양이, TypeScript)
   - **Action/verb noun**: what would someone ask? (사는곳, 마시는것, 키우는것, 쓰는언어)
   - **Colloquial variants**: casual phrasing (집, 좋아하는거, 펫, 코딩)
   - **Synonyms**: alternative expressions (주소→위치, 음료→마실것, 반려동물→애완동물)

   ✅ 올바른 예시:
   - "서울 성수동에 산다" → keys: ["거주지", "성수동", "서울", "사는곳", "집", "주소", "위치"]
   - "고양이 두 마리 키운다" → keys: ["반려동물", "고양이", "키우는것", "펫", "동물", "애완동물"]
   - "아이스 아메리카노 매일 마심" → keys: ["음료", "커피", "아메리카노", "마시는것", "취향", "즐겨마심"]
   - "TypeScript 주력 사용" → keys: ["언어", "TypeScript", "개발언어", "코딩", "쓰는언어", "프로그래밍"]

   ❌ 나쁜 예시 (너무 formal/좁음):
   - "서울 성수동에 산다" → keys: ["거주지", "성수동"] ← "어디 살아" 검색 시 못 찾음
   - "고양이 키운다" → keys: ["고양이", "반려동물"] ← "키우는거 있어?" 검색 시 못 찾음

9. **Names only as keys for identity memories.**
   - "사용자 이름은 동건" → keys: ["이름", "사용자", "동건"]
   - "좋아하는 과일은 딸기" → keys: ["과일", "딸기", "좋아함", "취향"] ← no name
10. Set \`key_types\` for names/proper nouns:
    - \`"name"\`: exact match only. \`"proper_noun"\`: exact match only.
    Example: key_types: {{"동건": "name"}}

### Correct
11. Use \`correct()\` when info changes. Never use \`remember()\` for updates.

### Explore
12. \`recall\` does 2-hop associative search automatically.
13. Use \`related()\` for deeper exploration after recall.

### Delete
14. \`forget()\` only for completely wrong information. For outdated info, use \`correct()\`.
`;

export const graph = new MemoryGraph();

function stats(): string {
  return `${Object.keys(graph.keys).length} keys, ${graph.listAll().length} memories, ${graph.linkCount} links`;
}

export const server = new Server(
  { name: "super-memory", version: "0.4.4" },
  { capabilities: { tools: {}, prompts: {} } }
);

// ── Tool definitions ──

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "recall",
      description:
        "CALL THIS FIRST before every first response. Search long-term memory by concept. namespace filters to a specific project/context. expand=True returns up to 2x results by following explicit memory links — use when initial results feel insufficient. Returns memories ranked by relevance with hop=1 (direct) or hop=2 (associative). Memories get stronger each time they're recalled.",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string" },
          top_k: { type: "number" },
          namespace: { type: "string" },
          expand: { type: "boolean" },
        },
        required: ["query"],
      },
    },
    {
      name: "remember",
      description:
        "Save important information to memory. Keys are search terms — think 'what would I search to find this later?' Use 3-6 diverse keys. namespace groups memories by project/context (e.g. 'work', 'personal'). ttl_seconds sets expiry for temporary memories (e.g. 3600 = 1 hour; None = permanent). related_to links this memory to existing memory IDs for explicit graph traversal.",
      inputSchema: {
        type: "object",
        properties: {
          content: { type: "string" },
          keys: { type: "array", items: { type: "string" } },
          key_types: {
            type: "object",
            additionalProperties: { type: "string" },
          },
          namespace: { type: "string" },
          ttl_seconds: { type: "number" },
          related_to: { type: "array", items: { type: "string" } },
        },
        required: ["content", "keys"],
      },
    },
    {
      name: "correct",
      description:
        "Update outdated information. Use when user corrects you or info changes (e.g. moved cities, changed job). Old version is preserved but weakened — never lost. Omit keys to keep the same search terms. related_to links the updated memory to other memory IDs.",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
          content: { type: "string" },
          keys: { type: "array", items: { type: "string" } },
          key_types: {
            type: "object",
            additionalProperties: { type: "string" },
          },
          related_to: { type: "array", items: { type: "string" } },
        },
        required: ["memory_id", "content"],
      },
    },
    {
      name: "related",
      description:
        "Explore connections from a specific memory. Returns memories connected by shared keys OR explicit links (both directions). Use after recall() to drill down: recall → pick ID → related → pick ID → related → ...",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
        },
        required: ["memory_id"],
      },
    },
    {
      name: "forget",
      description:
        "Permanently delete a memory. Only use for completely wrong information. For outdated info, use correct() instead — it preserves history.",
      inputSchema: {
        type: "object",
        properties: {
          memory_id: { type: "string" },
        },
        required: ["memory_id"],
      },
    },
    {
      name: "get_conversation",
      description:
        "Load raw conversation turns from a past session. Use when a recalled memory lacks detail and you need the original context.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: { type: "string" },
          turn: { type: "number" },
        },
        required: ["session_id"],
      },
    },
    {
      name: "list_memories",
      description:
        "List all stored memories. namespace filters by project/context. Expired memories are excluded. Prefer recall() for normal retrieval.",
      inputSchema: {
        type: "object",
        properties: {
          namespace: { type: "string" },
        },
        required: [],
      },
    },
    {
      name: "remember_batch",
      description:
        "Save multiple memories in one call. Each item: {content, keys, key_types?, namespace?, ttl_seconds?, related_to?}. Returns list of saved IDs. More efficient than multiple remember() calls.",
      inputSchema: {
        type: "object",
        properties: {
          items: {
            type: "array",
            items: {
              type: "object",
              properties: {
                content: { type: "string" },
                keys: { type: "array", items: { type: "string" } },
                key_types: {
                  type: "object",
                  additionalProperties: { type: "string" },
                },
                namespace: { type: "string" },
                ttl_seconds: { type: "number" },
                related_to: { type: "array", items: { type: "string" } },
              },
              required: ["content", "keys"],
            },
          },
        },
        required: ["items"],
      },
    },
    {
      name: "cleanup_expired",
      description:
        "Delete all memories past their ttl. Returns count of deleted memories. Call periodically to keep memory clean.",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
    {
      name: "memory_stats",
      description: "Get counts of keys, memories, and links in the system.",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  ],
}));

// ── Tool call handler ──

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const a = (args ?? {}) as Record<string, unknown>;

  try {
    switch (name) {
      case "recall": {
        const results = await graph.recall(
          a.query as string,
          typeof a.top_k === "number" ? a.top_k : 5,
          typeof a.namespace === "string" ? a.namespace : null,
          typeof a.expand === "boolean" ? a.expand : false
        );
        return { content: [{ type: "text", text: JSON.stringify(results, null, 0) }] };
      }

      case "remember": {
        const keys = sanitizeKeys(a.keys);
        const [mid, wasDedup] = await graph.add(
          a.content as string,
          keys,
          {
            keyTypes: parseObject(a.key_types) as Record<string, string> | null,
            namespace: typeof a.namespace === "string" ? a.namespace : "default",
            ttlSeconds: typeof a.ttl_seconds === "number" ? a.ttl_seconds : null,
            relatedTo: parseArray(a.related_to) as string[] | null,
          }
        );
        const result = wasDedup
          ? { saved: mid, deduplicated: true, note: "Similar memory existed — updated instead of creating duplicate" }
          : { saved: mid };
        return { content: [{ type: "text", text: JSON.stringify(result) }] };
      }

      case "correct": {
        const nid = await graph.supersede(
          a.memory_id as string,
          a.content as string,
          {
            keyConcepts: parseArray(a.keys) as string[] | null,
            keyTypes: parseObject(a.key_types) as Record<string, string> | null,
            relatedTo: parseArray(a.related_to) as string[] | null,
          }
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ new_id: nid, superseded: a.memory_id }),
            },
          ],
        };
      }

      case "related": {
        const results = graph.getRelated(a.memory_id as string);
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "forget": {
        const ok = await graph.delete(a.memory_id as string);
        return { content: [{ type: "text", text: JSON.stringify({ deleted: ok }) }] };
      }

      case "get_conversation": {
        const turns = await loadConversation(
          a.session_id as string,
          typeof a.turn === "number" ? a.turn : null
        );
        return { content: [{ type: "text", text: JSON.stringify(turns) }] };
      }

      case "list_memories": {
        const results = graph.listAll(
          typeof a.namespace === "string" ? a.namespace : null
        );
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "remember_batch": {
        const items = (parseArray(a.items) ?? []) as Array<Record<string, unknown>>;
        const results: object[] = [];
        for (const item of items) {
          const content = item.content as string;
          const keys = sanitizeKeys(item.keys);
          if (!content || keys.length === 0) {
            results.push({ error: "content and keys required", item });
            continue;
          }
          const [mid, wasDedup] = await graph.add(content, keys, {
            keyTypes: item.key_types as Record<string, string> | null,
            namespace: typeof item.namespace === "string" ? item.namespace : "default",
            ttlSeconds: typeof item.ttl_seconds === "number" ? item.ttl_seconds : null,
            relatedTo: Array.isArray(item.related_to) ? (item.related_to as string[]) : null,
          });
          results.push({ saved: mid, deduplicated: wasDedup });
        }
        return { content: [{ type: "text", text: JSON.stringify(results) }] };
      }

      case "cleanup_expired": {
        const count = await graph.cleanupExpired();
        return { content: [{ type: "text", text: JSON.stringify({ deleted: count }) }] };
      }

      case "memory_stats": {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({
                keys: Object.keys(graph.keys).length,
                memories: graph.listAll().length,
                links: graph.linkCount,
              }),
            },
          ],
        };
      }

      default:
        return {
          content: [{ type: "text", text: JSON.stringify({ error: `Unknown tool: ${name}` }) }],
          isError: true,
        };
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return {
      content: [{ type: "text", text: JSON.stringify({ error: msg }) }],
      isError: true,
    };
  }
});

// ── Prompt definitions ──

server.setRequestHandler(ListPromptsRequestSchema, async () => ({
  prompts: [
    {
      name: "memory_system_prompt",
      description:
        "System prompt for LLM agents using super-memory. Include this in your system prompt.",
    },
  ],
}));

server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  if (request.params.name !== "memory_system_prompt") {
    throw new Error(`Unknown prompt: ${request.params.name}`);
  }
  return {
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: MEMORY_SYSTEM.replace("{stats}", stats()),
        },
      },
    ],
  };
});
