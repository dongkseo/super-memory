import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH_FILE = DATA_DIR / "graph.json"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
KEY_MERGE_THRESHOLD = 0.85  # OpenAI 임베딩은 유사도 분포가 더 정확
DEPTH_INCREMENT = 0.05
DEPTH_MAX = 1.0
DEPTH_DEEP_THRESHOLD = 0.7  # 이 이상이면 deep memory (correction 저항)

def embed_text(text: str) -> list[float]:
    """OpenAI Embedding API 호출 (동기, load 전용)."""
    resp = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": OPENAI_EMBEDDING_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


_async_client: httpx.AsyncClient | None = None


def _get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(timeout=30)
    return _async_client


async def embed_text_async(text: str) -> list[float]:
    """OpenAI Embedding API 호출 (비동기)."""
    client = _get_async_client()
    resp = await client.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": OPENAI_EMBEDDING_MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    dot = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ── Data Structures ──


@dataclass
class Key:
    id: str
    concept: str
    embedding: list[float]
    key_type: str = "concept"  # "concept" | "name" | "proper_noun"


@dataclass
class Memory:
    id: str
    content: str
    embedding: list[float]  # content 임베딩 (직접 매칭용)
    created_at: float
    source: dict | None = None
    supersedes: str | None = None
    depth: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0


@dataclass
class Link:
    key_id: str
    memory_id: str


# ── Graph ──


@dataclass
class MemoryGraph:
    keys: dict[str, Key] = field(default_factory=dict)
    memories: dict[str, Memory] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    _dirty: bool = field(default=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def load(self) -> None:
        raw = _read_json(GRAPH_FILE)
        if not raw:
            return
        for kid, k in raw.get("keys", {}).items():
            self.keys[kid] = Key(**k)
        for mid, m in raw.get("memories", {}).items():
            if "embedding" not in m:
                m["embedding"] = embed_text(m["content"])
            self.memories[mid] = Memory(**m)
        for lnk in raw.get("links", []):
            self.links.append(Link(**lnk))
        print(f"[graph] loaded {len(self.keys)} keys, {len(self.memories)} memories, {len(self.links)} links")

    async def save(self) -> None:
        """비동기로 디스크에 저장하고 dirty 플래그 해제."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "keys": {kid: _key_dict(k) for kid, k in self.keys.items()},
            "memories": {mid: _mem_dict(m) for mid, m in self.memories.items()},
            "links": [{"key_id": l.key_id, "memory_id": l.memory_id} for l in self.links],
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        await asyncio.to_thread(GRAPH_FILE.write_text, content)
        self._dirty = False

    def mark_dirty(self) -> None:
        """변경이 있음을 표시. flush()로 나중에 저장."""
        self._dirty = True

    async def flush(self) -> None:
        """dirty 상태일 때만 저장."""
        if self._dirty:
            await self.save()

    # ── Key 관리 ──

    async def find_or_create_key(self, concept: str, key_type: str = "concept") -> str:
        """의미적으로 유사한 기존 키가 있으면 재사용, 없으면 생성."""
        # name/proper_noun 키는 정확 매칭만
        if key_type in ("name", "proper_noun"):
            for kid, key in self.keys.items():
                if key.concept == concept and key.key_type == key_type:
                    return kid
            kid = _uid()
            self.keys[kid] = Key(id=kid, concept=concept, embedding=await embed_text_async(concept), key_type=key_type)
            return kid

        # concept 키는 임베딩 유사도로 병합
        emb = await embed_text_async(concept)
        best_sim, best_kid = 0.0, None
        for kid, key in self.keys.items():
            if key.key_type != "concept":
                continue
            sim = cosine_sim(emb, key.embedding)
            if sim > best_sim:
                best_sim = sim
                best_kid = kid
        if best_sim >= KEY_MERGE_THRESHOLD and best_kid:
            return best_kid
        kid = _uid()
        self.keys[kid] = Key(id=kid, concept=concept, embedding=emb, key_type="concept")
        return kid

    def get_keys_for_memory(self, memory_id: str) -> list[str]:
        """메모리에 연결된 키 concept 목록."""
        return [
            self.keys[l.key_id].concept
            for l in self.links
            if l.memory_id == memory_id and l.key_id in self.keys
        ]

    def _key_idf(self, key_id: str) -> float:
        """IDF: 많은 기억에 연결된 키일수록 가중치 낮춤."""
        freq = sum(1 for l in self.links if l.key_id == key_id)
        if freq <= 1:
            return 1.0
        # 2개 연결이면 0.5, 3개면 0.33, ... (1/freq)
        idf = 1.0 / freq
        # name/proper_noun이 허브가 되면 추가 감쇠
        if key_id in self.keys and self.keys[key_id].key_type in ("name", "proper_noun"):
            idf *= 0.5
        return idf

    # ── 기억 추가 ──

    async def add(self, content: str, key_concepts: list[str],
            key_types: dict[str, str] | None = None,
            source: dict | None = None) -> str:
        """key_types: {"동건": "name", "파이썬": "concept"} 형태로 타입 지정."""
        async with self._lock:
            key_types = key_types or {}
            mid = _uid()
            self.memories[mid] = Memory(
                id=mid,
                content=content,
                embedding=await embed_text_async(content),
                created_at=time.time(),
                source=source,
                last_accessed=time.time(),
            )
            key_concepts = _sanitize_keys(key_concepts)
            for concept in key_concepts:
                kt = key_types.get(concept, "concept")
                kid = await self.find_or_create_key(concept, key_type=kt)
                if not any(l.key_id == kid and l.memory_id == mid for l in self.links):
                    self.links.append(Link(key_id=kid, memory_id=mid))
            await self.save()
            return mid

    # ── 기억 수정 (supersede) ──

    async def supersede(
        self, old_id: str, new_content: str,
        key_concepts: list[str] | None = None,
        key_types: dict[str, str] | None = None,
        source: dict | None = None,
    ) -> str:
        async with self._lock:
            if old_id not in self.memories:
                raise ValueError(f"Memory {old_id} not found")

            old = self.memories[old_id]
            if old.depth >= DEPTH_DEEP_THRESHOLD:
                old.depth *= 0.8
            else:
                old.depth *= 0.3

            key_types = key_types or {}
            mid = _uid()
            self.memories[mid] = Memory(
                id=mid,
                content=new_content,
                embedding=await embed_text_async(new_content),
                created_at=time.time(),
                source=source,
                supersedes=old_id,
                last_accessed=time.time(),
            )

            if key_concepts:
                key_concepts = _sanitize_keys(key_concepts)
                for concept in key_concepts:
                    kt = key_types.get(concept, "concept")
                    kid = await self.find_or_create_key(concept, key_type=kt)
                    self.links.append(Link(key_id=kid, memory_id=mid))
            else:
                for l in self.links:
                    if l.memory_id == old_id:
                        self.links.append(Link(key_id=l.key_id, memory_id=mid))

            await self.save()
            return mid

    # ── N:M 검색 (멀티홉 + IDF) ──

    HOP_DECAY = 0.3
    TIME_HALF_LIFE = 30 * 24 * 3600  # 30일 반감기

    def _time_factor(self, mem: "Memory") -> float:
        """시간 decay. deep 기억은 거의 안 잊혀지고 shallow는 빠르게 사라짐."""
        age = time.time() - mem.created_at
        decay_rate = 1.0 - mem.depth * 0.7  # deep(1.0)이면 0.3, shallow(0.0)이면 1.0
        decay = math.exp(-age * decay_rate / self.TIME_HALF_LIFE)
        return 0.5 + 0.5 * decay  # 0.5 ~ 1.0 범위

    async def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """두 경로로 검색: (A) 키 매칭 → 링크 → 기억, (B) content 직접 매칭."""
        if not self.memories:
            return []

        q_emb = await embed_text_async(query)

        async with self._lock:
            query_lower = query.lower().strip()

            mem_scores: dict[str, float] = {}
            mem_matched_keys: dict[str, list[str]] = {}
            mem_hop: dict[str, int] = {}

            # ── 경로 A: 키 매칭 → 링크 → 기억 ──
            key_scores: list[tuple[float, str]] = []
            for kid, key in self.keys.items():
                if key.key_type in ("name", "proper_noun"):
                    if key.concept.lower() in query_lower:
                        key_scores.append((1.0, kid))
                else:
                    sim = cosine_sim(q_emb, key.embedding)
                    if sim >= 0.35:
                        key_scores.append((sim, kid))
            key_scores.sort(reverse=True)

            for key_sim, kid in key_scores[:10]:
                idf = self._key_idf(kid)
                for l in self.links:
                    if l.key_id != kid or l.memory_id not in self.memories:
                        continue
                    mid = l.memory_id
                    mem = self.memories[mid]
                    depth_factor = 0.5 + mem.depth * 0.5
                    tf = self._time_factor(mem)
                    score = key_sim * idf * depth_factor * tf
                    mem_scores[mid] = mem_scores.get(mid, 0) + score
                    mem_matched_keys.setdefault(mid, []).append(self.keys[kid].concept)
                    mem_hop[mid] = 1

            # ── 경로 B: content 직접 매칭 ──
            for mid, mem in self.memories.items():
                content_sim = cosine_sim(q_emb, mem.embedding)
                if content_sim >= 0.3:
                    depth_factor = 0.5 + mem.depth * 0.5
                    tf = self._time_factor(mem)
                    content_score = content_sim * depth_factor * tf
                    mem_scores[mid] = mem_scores.get(mid, 0) + content_score
                    mem_matched_keys.setdefault(mid, []).append("(content)")
                    if mid not in mem_hop:
                        mem_hop[mid] = 1

            # 3단계 (2홉): 1홉 기억의 다른 키 → 새로운 기억
            hop1_mids = set(mem_scores.keys())
            for mid in list(hop1_mids):
                hop1_score = mem_scores[mid]
                my_kids = {l.key_id for l in self.links if l.memory_id == mid}
                for kid in my_kids:
                    if kid not in self.keys:
                        continue
                    concept = self.keys[kid].concept
                    idf = self._key_idf(kid)
                    for l in self.links:
                        if l.key_id != kid or l.memory_id == mid or l.memory_id not in self.memories:
                            continue
                        other_mid = l.memory_id
                        hop2_score = hop1_score * self.HOP_DECAY * idf
                        mem_scores[other_mid] = mem_scores.get(other_mid, 0) + hop2_score
                        mem_matched_keys.setdefault(other_mid, []).append(f"{concept}(via)")
                        if other_mid not in mem_hop:
                            mem_hop[other_mid] = 2

            # 4단계: 정렬, depth 강화, 반환
            ranked = sorted(mem_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for mid, score in ranked:
                mem = self.memories[mid]
                mem.depth = min(mem.depth + DEPTH_INCREMENT, DEPTH_MAX)
                mem.access_count += 1
                mem.last_accessed = time.time()

                superseded_by = None
                for m in self.memories.values():
                    if m.supersedes == mid:
                        superseded_by = m.id
                        break

                results.append({
                    "id": mid,
                    "content": mem.content,
                    "keys": self.get_keys_for_memory(mid),
                    "matched_via": list(dict.fromkeys(mem_matched_keys.get(mid, []))),
                    "hop": mem_hop.get(mid, 1),
                    "score": round(score, 3),
                    "depth": round(mem.depth, 3),
                    "access_count": mem.access_count,
                    "source": mem.source,
                    "supersedes": mem.supersedes,
                    "superseded_by": superseded_by,
                    "created_at": mem.created_at,
                })

            self.mark_dirty()
            await self.flush()
            return results

    # ── 연관 기억 탐색 (공유 키 기반) ──

    def get_related(self, memory_id: str) -> list[dict]:
        """이 기억의 키를 공유하는 다른 기억 찾기 = 연상."""
        if memory_id not in self.memories:
            return []

        my_kids = {l.key_id for l in self.links if l.memory_id == memory_id}

        related: dict[str, dict] = {}
        for kid in my_kids:
            concept = self.keys[kid].concept if kid in self.keys else "?"
            for l in self.links:
                if l.key_id == kid and l.memory_id != memory_id and l.memory_id in self.memories:
                    mid = l.memory_id
                    mem = self.memories[mid]
                    if mid not in related:
                        related[mid] = {
                            "id": mid,
                            "content": mem.content,
                            "shared_keys": [],
                            "depth": round(mem.depth, 3),
                        }
                    if concept not in related[mid]["shared_keys"]:
                        related[mid]["shared_keys"].append(concept)

        return list(related.values())

    # ── 삭제 ──

    async def delete(self, memory_id: str) -> bool:
        async with self._lock:
            if memory_id not in self.memories:
                return False
            del self.memories[memory_id]
            self.links = [l for l in self.links if l.memory_id != memory_id]
            used_kids = {l.key_id for l in self.links}
            self.keys = {kid: k for kid, k in self.keys.items() if kid in used_kids}
            await self.save()
            return True

    # ── 전체 조회 ──

    def list_all(self) -> list[dict]:
        results = []
        for mid, mem in self.memories.items():
            results.append({
                "id": mid,
                "content": mem.content,
                "keys": self.get_keys_for_memory(mid),
                "depth": round(mem.depth, 3),
                "access_count": mem.access_count,
                "supersedes": mem.supersedes,
                "created_at": mem.created_at,
            })
        return results


# ── 대화 원본 ──


def save_turn(session_id: str, role: str, content: str) -> int:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONVERSATIONS_DIR / f"{session_id}.jsonl"
    turn = 0
    if path.exists():
        turn = sum(1 for _ in open(path, encoding="utf-8"))
    entry = json.dumps({"turn": turn, "role": role, "content": content, "ts": time.time()}, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
    return turn


def load_conversation(session_id: str, turn: int | None = None) -> list[dict]:
    path = CONVERSATIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    lines = [json.loads(line) for line in open(path, encoding="utf-8")]
    if turn is not None:
        start = max(0, turn - 2)
        end = min(len(lines), turn + 3)
        return lines[start:end]
    return lines


# ── Utils ──


def _sanitize_keys(keys: list) -> list[str]:
    """잘못된 키 필터링: 문자열이 아닌 것, 2글자 미만, JSON 파싱 시도."""
    if isinstance(keys, str):
        try:
            keys = json.loads(keys)
        except (json.JSONDecodeError, TypeError):
            keys = [keys]
    return [k.strip() for k in keys if isinstance(k, str) and len(k.strip()) >= 2]


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _key_dict(k: Key) -> dict:
    return {"id": k.id, "concept": k.concept, "embedding": k.embedding, "key_type": k.key_type}


def _mem_dict(m: Memory) -> dict:
    return {
        "id": m.id, "content": m.content, "embedding": m.embedding, "created_at": m.created_at,
        "source": m.source, "supersedes": m.supersedes,
        "depth": m.depth, "access_count": m.access_count,
        "last_accessed": m.last_accessed,
    }
