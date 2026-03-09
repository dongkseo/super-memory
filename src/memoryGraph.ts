import { readFile, writeFile, mkdir, appendFile } from "fs/promises";
import { randomBytes } from "crypto";
import { join } from "path";
import { homedir } from "os";
import { Mutex } from "async-mutex";
import { embedTextAsync, EMBEDDING_BACKEND } from "./embedding.js";
import type { Key, Memory, GraphData } from "./types.js";

const DATA_DIR =
  process.env.SUPER_MEMORY_DATA_DIR ?? join(homedir(), ".super-memory");
const GRAPH_FILE = join(DATA_DIR, "graph.json");
const CONVERSATIONS_DIR = join(DATA_DIR, "conversations");

const KEY_MERGE_THRESHOLD = 0.85;
const MEMORY_DEDUP_THRESHOLD = 0.9;
const KEY_AUTO_LINK_THRESHOLD = 0.5;
const DEPTH_INCREMENT = 0.05;
const DEPTH_MAX = 1.0;
const DEPTH_DEEP_THRESHOLD = 0.7;

// ── Vector math ──

function cosineSim(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const norm = Math.sqrt(normA) * Math.sqrt(normB);
  return norm === 0 ? 0 : dot / norm;
}

function batchCosineSim(query: number[], matrix: number[][]): number[] {
  if (matrix.length === 0) return [];
  return matrix.map((row) => cosineSim(query, row));
}

// ── Utils ──

function uid(): string {
  return randomBytes(6).toString("hex");
}

export function sanitizeKeys(keys: unknown): string[] {
  let arr: unknown[];
  if (typeof keys === "string") {
    try {
      arr = JSON.parse(keys);
    } catch {
      arr = [keys];
    }
  } else if (Array.isArray(keys)) {
    arr = keys;
  } else {
    return [];
  }
  return arr
    .filter((k): k is string => typeof k === "string" && k.trim().length >= 2)
    .map((k) => k.trim());
}

// ── MemoryGraph ──

export class MemoryGraph {
  keys: Record<string, Key> = {};
  memories: Record<string, Memory> = {};

  private _keyToMems: Record<string, Set<string>> = {};
  private _memToKeys: Record<string, Set<string>> = {};
  private _supersededBy: Record<string, string> = {};
  private _storedDim: number | null = null;
  private _lock = new Mutex();
  private _dirty = false;

  static readonly HOP_DECAY = 0.3;
  static readonly TIME_HALF_LIFE = 30 * 24 * 3600;

  get linkCount(): number {
    return Object.values(this._keyToMems).reduce(
      (sum, mids) => sum + mids.size,
      0
    );
  }

  private _link(keyId: string, memId: string): void {
    if (!this._keyToMems[keyId]) this._keyToMems[keyId] = new Set();
    this._keyToMems[keyId].add(memId);
    if (!this._memToKeys[memId]) this._memToKeys[memId] = new Set();
    this._memToKeys[memId].add(keyId);
  }

  private _hasLink(keyId: string, memId: string): boolean {
    return this._keyToMems[keyId]?.has(memId) ?? false;
  }

  private _unlinkMemory(memId: string): void {
    const kids = this._memToKeys[memId];
    if (kids) {
      for (const kid of kids) {
        const mems = this._keyToMems[kid];
        if (mems) {
          mems.delete(memId);
          if (mems.size === 0) delete this._keyToMems[kid];
        }
      }
      delete this._memToKeys[memId];
    }
  }

  private _pruneOrphanKeys(): void {
    for (const kid of Object.keys(this.keys)) {
      const mems = this._keyToMems[kid];
      if (!mems || mems.size === 0) delete this.keys[kid];
    }
  }

  private _checkDim(embedding: number[]): void {
    const dim = embedding.length;
    if (this._storedDim === null) {
      this._storedDim = dim;
      return;
    }
    if (dim !== this._storedDim) {
      throw new Error(
        `Embedding dimension mismatch: existing data uses ${this._storedDim}-dim, ` +
          `current backend (${EMBEDDING_BACKEND}) produces ${dim}-dim.\n` +
          `To switch backends, delete ~/.super-memory/graph.json first.`
      );
    }
  }

  private _isExpired(mem: Memory): boolean {
    return mem.ttl != null && Date.now() / 1000 > mem.ttl;
  }

  private _timeFactor(mem: Memory): number {
    const age = Date.now() / 1000 - mem.created_at;
    const decayRate = 1.0 - mem.depth * 0.7;
    const decay = Math.exp((-age * decayRate) / MemoryGraph.TIME_HALF_LIFE);
    return 0.5 + 0.5 * decay;
  }

  private _keyIdf(keyId: string): number {
    const freq = this._keyToMems[keyId]?.size ?? 0;
    if (freq <= 1) return 1.0;
    let idf = 1.0 / freq;
    const kt = this.keys[keyId]?.key_type;
    if (kt === "name" || kt === "proper_noun") idf *= 0.5;
    return idf;
  }

  private _findDuplicate(embedding: number[]): string | null {
    const activeMems = Object.entries(this.memories).filter(
      ([mid]) => !(mid in this._supersededBy)
    );
    if (activeMems.length === 0) return null;
    const matrix = activeMems.map(([, mem]) => mem.embedding);
    const sims = batchCosineSim(embedding, matrix);
    let bestIdx = 0,
      bestSim = -Infinity;
    for (let i = 0; i < sims.length; i++) {
      if (sims[i] > bestSim) {
        bestSim = sims[i];
        bestIdx = i;
      }
    }
    return bestSim >= MEMORY_DEDUP_THRESHOLD ? activeMems[bestIdx][0] : null;
  }

  private _autoLinkKeys(memId: string, embedding: number[]): void {
    const keyIds = Object.keys(this.keys);
    if (keyIds.length === 0) return;
    const matrix = keyIds.map((kid) => this.keys[kid].embedding);
    const sims = batchCosineSim(embedding, matrix);
    for (let i = 0; i < keyIds.length; i++) {
      if (sims[i] >= KEY_AUTO_LINK_THRESHOLD && !this._hasLink(keyIds[i], memId)) {
        this._link(keyIds[i], memId);
      }
    }
  }

  getKeysForMemory(memId: string): string[] {
    const kids = this._memToKeys[memId];
    if (!kids) return [];
    return [...kids]
      .filter((kid) => kid in this.keys)
      .map((kid) => this.keys[kid].concept);
  }

  // ── I/O ──

  async load(): Promise<void> {
    let raw: GraphData;
    try {
      const text = await readFile(GRAPH_FILE, "utf-8");
      raw = JSON.parse(text) as GraphData;
    } catch {
      return;
    }

    for (const [kid, k] of Object.entries(raw.keys ?? {})) {
      this.keys[kid] = k;
    }

    for (const [mid, m] of Object.entries(raw.memories ?? {})) {
      const defaults = {
        depth: 0.0,
        access_count: 0,
        last_accessed: 0,
        namespace: "default",
        ttl: null,
        links: [] as string[],
        source: null,
        supersedes: null,
      };
      const mem: Memory = { ...defaults, ...m };
      if (!mem.embedding || mem.embedding.length === 0) {
        mem.embedding = await embedTextAsync(mem.content);
      }
      this.memories[mid] = mem;
    }

    if (Object.keys(this.memories).length > 0) {
      const firstMem = Object.values(this.memories)[0];
      this._storedDim = firstMem.embedding.length;
    }

    for (const lnk of raw.links ?? []) {
      this._link(lnk.key_id, lnk.memory_id);
    }

    for (const [mid, mem] of Object.entries(this.memories)) {
      if (mem.supersedes) {
        this._supersededBy[mem.supersedes] = mid;
      }
    }

    console.error(
      `[graph] loaded ${Object.keys(this.keys).length} keys, ` +
        `${Object.keys(this.memories).length} memories, ${this.linkCount} links`
    );
  }

  async save(): Promise<void> {
    await mkdir(DATA_DIR, { recursive: true });
    const links: Array<{ key_id: string; memory_id: string }> = [];
    for (const [kid, mids] of Object.entries(this._keyToMems)) {
      for (const mid of mids) {
        links.push({ key_id: kid, memory_id: mid });
      }
    }
    const data: GraphData = {
      keys: this.keys,
      memories: this.memories,
      links,
    };
    await writeFile(GRAPH_FILE, JSON.stringify(data, null, 2), "utf-8");
    this._dirty = false;
  }

  markDirty(): void {
    this._dirty = true;
  }

  async flush(): Promise<void> {
    if (this._dirty) await this.save();
  }

  // ── Key management ──

  async findOrCreateKey(
    concept: string,
    keyType: "concept" | "name" | "proper_noun" = "concept"
  ): Promise<string> {
    if (keyType === "name" || keyType === "proper_noun") {
      for (const [kid, key] of Object.entries(this.keys)) {
        if (key.concept === concept && key.key_type === keyType) return kid;
      }
      const kid = uid();
      this.keys[kid] = {
        id: kid,
        concept,
        embedding: await embedTextAsync(concept),
        key_type: keyType,
      };
      return kid;
    }

    const emb = await embedTextAsync(concept);
    const conceptKeys = Object.entries(this.keys).filter(
      ([, k]) => k.key_type === "concept"
    );
    if (conceptKeys.length > 0) {
      const matrix = conceptKeys.map(([, k]) => k.embedding);
      const sims = batchCosineSim(emb, matrix);
      let bestIdx = 0,
        bestSim = -Infinity;
      for (let i = 0; i < sims.length; i++) {
        if (sims[i] > bestSim) {
          bestSim = sims[i];
          bestIdx = i;
        }
      }
      if (bestSim >= KEY_MERGE_THRESHOLD) return conceptKeys[bestIdx][0];
    }

    const kid = uid();
    this.keys[kid] = { id: kid, concept, embedding: emb, key_type: "concept" };
    return kid;
  }

  // ── Add ──

  async add(
    content: string,
    keyConcepts: string[],
    options: {
      keyTypes?: Record<string, string> | null;
      source?: Record<string, unknown> | null;
      namespace?: string;
      ttlSeconds?: number | null;
      relatedTo?: string[] | null;
    } = {}
  ): Promise<[string, boolean]> {
    const embedding = await embedTextAsync(content); // outside lock

    let dupId: string | null = null;
    await this._lock.runExclusive(async () => {
      this._checkDim(embedding);
      dupId = this._findDuplicate(embedding);
    });

    if (dupId !== null) {
      const newId = await this.supersede(dupId, content, {
        keyConcepts,
        keyTypes: options.keyTypes ?? undefined,
        source: options.source,
        namespace: options.namespace,
        relatedTo: options.relatedTo,
      });
      return [newId, true];
    }

    let resultMid = "";
    await this._lock.runExclusive(async () => {
      const mid = uid();
      resultMid = mid;
      const now = Date.now() / 1000;
      const expiresAt =
        options.ttlSeconds != null ? now + options.ttlSeconds : null;
      const validLinks = (options.relatedTo ?? []).filter(
        (lid) => lid in this.memories
      );

      this.memories[mid] = {
        id: mid,
        content,
        embedding,
        created_at: now,
        source: options.source ?? null,
        supersedes: null,
        depth: 0.0,
        access_count: 0,
        last_accessed: now,
        namespace: options.namespace ?? "default",
        ttl: expiresAt,
        links: validLinks,
      };

      const sanitized = sanitizeKeys(keyConcepts);
      const keyTypes = options.keyTypes ?? {};
      for (const concept of sanitized) {
        const kt = ((keyTypes[concept] ?? "concept") as
          | "concept"
          | "name"
          | "proper_noun");
        const kid = await this.findOrCreateKey(concept, kt);
        if (!this._hasLink(kid, mid)) this._link(kid, mid);
      }

      this._autoLinkKeys(mid, embedding);
      await this.save();
    });

    return [resultMid, false];
  }

  // ── Supersede ──

  async supersede(
    oldId: string,
    newContent: string,
    options: {
      keyConcepts?: string[] | null;
      keyTypes?: Record<string, string> | null;
      source?: Record<string, unknown> | null;
      namespace?: string | null;
      relatedTo?: string[] | null;
    } = {}
  ): Promise<string> {
    const newEmbedding = await embedTextAsync(newContent); // outside lock

    let resultMid = "";
    await this._lock.runExclusive(async () => {
      if (!(oldId in this.memories)) {
        throw new Error(`Memory ${oldId} not found`);
      }

      const old = this.memories[oldId];

      // Chain cleanup: keep depth max 1 (new → old; grandparent deleted)
      const grandparentId = old.supersedes;
      if (grandparentId && grandparentId in this.memories) {
        delete this.memories[grandparentId];
        this._unlinkMemory(grandparentId);
        delete this._supersededBy[grandparentId];
        this._pruneOrphanKeys();
      }

      const mid = uid();
      resultMid = mid;
      const now = Date.now() / 1000;
      const ns = options.namespace ?? old.namespace;
      const validLinks = (options.relatedTo ?? []).filter(
        (lid) => lid in this.memories
      );

      this.memories[mid] = {
        id: mid,
        content: newContent,
        embedding: newEmbedding,
        created_at: now,
        source: options.source ?? null,
        supersedes: oldId,
        depth: 0.0,
        access_count: 0,
        last_accessed: now,
        namespace: ns,
        ttl: old.ttl,
        links: validLinks,
      };

      // Weaken old memory depth
      old.depth =
        old.depth >= DEPTH_DEEP_THRESHOLD
          ? old.depth * 0.8
          : old.depth * 0.3;
      this._supersededBy[oldId] = mid;

      const keyConcepts = options.keyConcepts;
      if (keyConcepts && keyConcepts.length > 0) {
        const sanitized = sanitizeKeys(keyConcepts);
        const keyTypes = options.keyTypes ?? {};
        for (const concept of sanitized) {
          const kt = ((keyTypes[concept] ?? "concept") as
            | "concept"
            | "name"
            | "proper_noun");
          const kid = await this.findOrCreateKey(concept, kt);
          this._link(kid, mid);
        }
      } else {
        // Copy old links (snapshot to avoid mutation during iteration)
        for (const kid of [...(this._memToKeys[oldId] ?? new Set())]) {
          this._link(kid, mid);
        }
      }

      this._autoLinkKeys(mid, newEmbedding);
      await this.save();
    });

    return resultMid;
  }

  // ── Recall ──

  async recall(
    query: string,
    topK = 5,
    namespace?: string | null,
    expand = false
  ): Promise<object[]> {
    if (Object.keys(this.memories).length === 0) return [];

    const qEmb = await embedTextAsync(query); // outside lock
    this._checkDim(qEmb);

    const results: object[] = [];

    await this._lock.runExclusive(async () => {
      const queryLower = query.toLowerCase().trim();
      const memScores: Record<string, number> = {};
      const memMatchedKeys: Record<string, string[]> = {};
      const memHop: Record<string, number> = {};

      const skip = (mid: string): boolean => {
        if (!(mid in this.memories)) return true;
        const mem = this.memories[mid];
        if (this._isExpired(mem)) return true;
        if (namespace && mem.namespace !== namespace) return true;
        if (mid in this._supersededBy) return true;
        return false;
      };

      // ── Path A: Key batch matching → links → memories ──
      const keyIds = Object.keys(this.keys);
      const keySims =
        keyIds.length > 0
          ? batchCosineSim(
              qEmb,
              keyIds.map((kid) => this.keys[kid].embedding)
            )
          : [];

      const keyScores: [number, string][] = [];
      for (let i = 0; i < keyIds.length; i++) {
        const kid = keyIds[i];
        const key = this.keys[kid];
        if (key.key_type === "name" || key.key_type === "proper_noun") {
          if (queryLower.includes(key.concept.toLowerCase())) {
            keyScores.push([1.0, kid]);
          }
        } else if (keySims[i] >= 0.35) {
          keyScores.push([keySims[i], kid]);
        }
      }
      keyScores.sort((a, b) => b[0] - a[0]);

      for (const [keySim, kid] of keyScores.slice(0, 10)) {
        const idf = this._keyIdf(kid);
        for (const memId of this._keyToMems[kid] ?? new Set()) {
          if (skip(memId)) continue;
          const mem = this.memories[memId];
          const depthFactor = 0.9 + mem.depth * 0.1;
          const tf = this._timeFactor(mem);
          const score = keySim * idf * depthFactor * tf;
          memScores[memId] = (memScores[memId] ?? 0) + score;
          if (!memMatchedKeys[memId]) memMatchedKeys[memId] = [];
          memMatchedKeys[memId].push(this.keys[kid].concept);
          memHop[memId] = 1;
        }
      }

      // ── Path B: Content batch direct matching ──
      const memIds = Object.keys(this.memories);
      if (memIds.length > 0) {
        const contentSims = batchCosineSim(
          qEmb,
          memIds.map((mid) => this.memories[mid].embedding)
        );
        for (let i = 0; i < memIds.length; i++) {
          const mid = memIds[i];
          if (skip(mid)) continue;
          const cSim = contentSims[i];
          if (cSim >= 0.35) {
            const mem = this.memories[mid];
            const depthFactor = 0.9 + mem.depth * 0.1;
            const tf = this._timeFactor(mem);
            const contentScore = cSim * depthFactor * tf * 0.8;
            if (mid in memScores) {
              memScores[mid] += contentScore * 0.2;
            } else {
              memScores[mid] = contentScore;
            }
            if (!memMatchedKeys[mid]) memMatchedKeys[mid] = [];
            memMatchedKeys[mid].push("(content)");
            if (!(mid in memHop)) memHop[mid] = 1;
          }
        }
      }

      // ── 2-hop: via shared keys ──
      for (const mid of Object.keys(memScores)) {
        const hop1Score = memScores[mid];
        for (const kid of this._memToKeys[mid] ?? new Set()) {
          if (!(kid in this.keys)) continue;
          const concept = this.keys[kid].concept;
          const idf = this._keyIdf(kid);
          for (const otherMid of this._keyToMems[kid] ?? new Set()) {
            if (otherMid === mid || skip(otherMid)) continue;
            const hop2Score = hop1Score * MemoryGraph.HOP_DECAY * idf;
            memScores[otherMid] = (memScores[otherMid] ?? 0) + hop2Score;
            if (!memMatchedKeys[otherMid]) memMatchedKeys[otherMid] = [];
            memMatchedKeys[otherMid].push(`${concept}(via)`);
            if (!(otherMid in memHop)) memHop[otherMid] = 2;
          }
        }
      }

      // ── Explicit link traversal ──
      for (const mid of Object.keys(memScores)) {
        const hop1Score = memScores[mid];
        const memObj = this.memories[mid];
        if (!memObj) continue;
        for (const linkedId of memObj.links) {
          if (linkedId === mid || skip(linkedId)) continue;
          const linkScore = hop1Score * MemoryGraph.HOP_DECAY;
          memScores[linkedId] = (memScores[linkedId] ?? 0) + linkScore;
          if (!memMatchedKeys[linkedId]) memMatchedKeys[linkedId] = [];
          memMatchedKeys[linkedId].push("(linked)");
          if (!(linkedId in memHop)) memHop[linkedId] = 2;
        }
      }

      if (expand) {
        for (const mid of Object.keys(memScores)) {
          if ((memHop[mid] ?? 1) === 2) memScores[mid] *= 0.7;
        }
      }

      const actualTopK = expand ? topK * 2 : topK;
      const ranked = Object.entries(memScores)
        .sort(([, a], [, b]) => b - a)
        .slice(0, actualTopK);

      for (const [mid, score] of ranked) {
        const mem = this.memories[mid];
        mem.depth = Math.min(mem.depth + DEPTH_INCREMENT, DEPTH_MAX);
        mem.access_count += 1;
        mem.last_accessed = Date.now() / 1000;
        results.push({
          id: mid,
          content: mem.content,
          keys: this.getKeysForMemory(mid),
          matched_via: [...new Set(memMatchedKeys[mid] ?? [])],
          hop: memHop[mid] ?? 1,
          score: Math.round(score * 1000) / 1000,
          depth: Math.round(mem.depth * 1000) / 1000,
          access_count: mem.access_count,
          source: mem.source,
          supersedes: mem.supersedes,
          superseded_by: this._supersededBy[mid] ?? null,
          created_at: mem.created_at,
          namespace: mem.namespace,
          links: mem.links,
        });
      }

      this.markDirty();
    });

    await this.flush(); // outside lock
    return results;
  }

  // ── Related ──

  getRelated(memoryId: string): object[] {
    if (!(memoryId in this.memories)) return [];

    const related: Record<
      string,
      {
        id: string;
        content: string;
        shared_keys: string[];
        link_type: string;
        depth: number;
      }
    > = {};

    // Key-sharing
    for (const kid of this._memToKeys[memoryId] ?? new Set()) {
      const concept = this.keys[kid]?.concept ?? "?";
      for (const mid of this._keyToMems[kid] ?? new Set()) {
        if (mid === memoryId || !(mid in this.memories)) continue;
        const mem = this.memories[mid];
        if (this._isExpired(mem) || mid in this._supersededBy) continue;
        if (!related[mid]) {
          related[mid] = {
            id: mid,
            content: mem.content,
            shared_keys: [],
            link_type: "key",
            depth: Math.round(mem.depth * 1000) / 1000,
          };
        }
        if (!related[mid].shared_keys.includes(concept)) {
          related[mid].shared_keys.push(concept);
        }
      }
    }

    // Explicit links (→)
    const sourceMem = this.memories[memoryId];
    for (const linkedId of sourceMem.links) {
      if (!(linkedId in this.memories) || linkedId === memoryId) continue;
      const mem = this.memories[linkedId];
      if (this._isExpired(mem)) continue;
      if (!related[linkedId]) {
        related[linkedId] = {
          id: linkedId,
          content: mem.content,
          shared_keys: ["(explicit →)"],
          link_type: "explicit",
          depth: Math.round(mem.depth * 1000) / 1000,
        };
      } else {
        related[linkedId].link_type = "both";
        if (!related[linkedId].shared_keys.includes("(explicit →)")) {
          related[linkedId].shared_keys.push("(explicit →)");
        }
      }
    }

    // Reverse links (←)
    for (const [mid, mem] of Object.entries(this.memories)) {
      if (mid === memoryId || this._isExpired(mem)) continue;
      if (mem.links.includes(memoryId)) {
        if (!related[mid]) {
          related[mid] = {
            id: mid,
            content: mem.content,
            shared_keys: ["(explicit ←)"],
            link_type: "explicit",
            depth: Math.round(mem.depth * 1000) / 1000,
          };
        } else if (!related[mid].shared_keys.includes("(explicit ←)")) {
          related[mid].shared_keys.push("(explicit ←)");
        }
      }
    }

    return Object.values(related);
  }

  // ── Delete ──

  async delete(memoryId: string): Promise<boolean> {
    return this._lock.runExclusive(async () => {
      if (!(memoryId in this.memories)) return false;
      delete this.memories[memoryId];
      this._unlinkMemory(memoryId);
      this._pruneOrphanKeys();
      delete this._supersededBy[memoryId];
      for (const [oldId, newId] of Object.entries(this._supersededBy)) {
        if (newId === memoryId) delete this._supersededBy[oldId];
      }
      await this.save();
      return true;
    });
  }

  // ── List all ──

  listAll(namespace?: string | null): object[] {
    return Object.entries(this.memories)
      .filter(([mid, mem]) => {
        if (this._isExpired(mem)) return false;
        if (mid in this._supersededBy) return false;
        if (namespace && mem.namespace !== namespace) return false;
        return true;
      })
      .map(([mid, mem]) => ({
        id: mid,
        content: mem.content,
        keys: this.getKeysForMemory(mid),
        depth: Math.round(mem.depth * 1000) / 1000,
        access_count: mem.access_count,
        supersedes: mem.supersedes,
        created_at: mem.created_at,
        namespace: mem.namespace,
        expires_at: mem.ttl,
        links: mem.links,
      }));
  }

  // ── Cleanup expired ──

  async cleanupExpired(): Promise<number> {
    return this._lock.runExclusive(async () => {
      const expired = Object.entries(this.memories)
        .filter(([, mem]) => this._isExpired(mem))
        .map(([mid]) => mid);

      for (const mid of expired) {
        delete this.memories[mid];
        this._unlinkMemory(mid);
        delete this._supersededBy[mid];
        for (const [oldId, newId] of Object.entries(this._supersededBy)) {
          if (newId === mid) delete this._supersededBy[oldId];
        }
      }
      this._pruneOrphanKeys();
      if (expired.length > 0) await this.save();
      return expired.length;
    });
  }
}

// ── Conversation store ──

export async function saveTurn(
  sessionId: string,
  role: string,
  content: string
): Promise<number> {
  await mkdir(CONVERSATIONS_DIR, { recursive: true });
  const path = join(CONVERSATIONS_DIR, `${sessionId}.jsonl`);
  let turn = 0;
  try {
    const text = await readFile(path, "utf-8");
    turn = text.split("\n").filter((l) => l.trim()).length;
  } catch {
    // file does not exist yet
  }
  const entry = JSON.stringify({
    turn,
    role,
    content,
    ts: Date.now() / 1000,
  });
  await appendFile(path, entry + "\n", "utf-8");
  return turn;
}

export async function loadConversation(
  sessionId: string,
  turn?: number | null
): Promise<object[]> {
  const path = join(CONVERSATIONS_DIR, `${sessionId}.jsonl`);
  let text: string;
  try {
    text = await readFile(path, "utf-8");
  } catch {
    return [];
  }
  const lines = text
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as object);
  if (turn != null) {
    const start = Math.max(0, turn - 2);
    const end = Math.min(lines.length, turn + 3);
    return lines.slice(start, end);
  }
  return lines;
}
