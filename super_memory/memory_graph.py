import asyncio
import atexit
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

# DATA_DIR: 환경변수 우선, 기본값은 ~/.super-memory/
DATA_DIR = Path(os.getenv("SUPER_MEMORY_DATA_DIR", Path.home() / ".super-memory"))
GRAPH_FILE = DATA_DIR / "graph.json"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
KEY_MERGE_THRESHOLD = 0.85
DEPTH_INCREMENT = 0.05
DEPTH_MAX = 1.0
DEPTH_DEEP_THRESHOLD = 0.7

_EMBED_RETRIES = 3


def embed_text(text: str) -> list[float]:
    """OpenAI Embedding API 호출 (동기, load 전용)."""
    for attempt in range(_EMBED_RETRIES):
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": OPENAI_EMBEDDING_MODEL, "input": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except (httpx.HTTPStatusError, httpx.TransportError):
            if attempt == _EMBED_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


_async_client: httpx.AsyncClient | None = None


def _get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(timeout=30)
    return _async_client


def _shutdown_async_client() -> None:
    """프로세스 종료 시 AsyncClient 정리."""
    global _async_client
    if _async_client and not _async_client.is_closed:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(_async_client.aclose())
        except Exception:
            pass


atexit.register(_shutdown_async_client)


async def embed_text_async(text: str) -> list[float]:
    """OpenAI Embedding API 호출 (비동기, retry 포함)."""
    client = _get_async_client()
    for attempt in range(_EMBED_RETRIES):
        try:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": OPENAI_EMBEDDING_MODEL, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except (httpx.HTTPStatusError, httpx.TransportError):
            if attempt == _EMBED_RETRIES - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _batch_cosine_sim(query_emb: list[float], matrix: np.ndarray) -> np.ndarray:
    """query_emb vs rows of matrix → shape (n,) cosine similarities."""
    if matrix.shape[0] == 0:
        return np.array([])
    q = np.array(query_emb)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    return np.where(norms == 0, 0.0, matrix @ q / norms)


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


# ── Graph ──


@dataclass
class MemoryGraph:
    keys: dict[str, Key] = field(default_factory=dict)
    memories: dict[str, Memory] = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        # 양방향 링크 인덱스: O(1) 조회
        self._key_to_mems: dict[str, set[str]] = {}
        self._mem_to_keys: dict[str, set[str]] = {}
        # supersede 역방향 인덱스: old_id → new_id
        self._superseded_by: dict[str, str] = {}

    # ── 링크 인덱스 헬퍼 ──

    def _link(self, key_id: str, memory_id: str) -> None:
        self._key_to_mems.setdefault(key_id, set()).add(memory_id)
        self._mem_to_keys.setdefault(memory_id, set()).add(key_id)

    def _has_link(self, key_id: str, memory_id: str) -> bool:
        return memory_id in self._key_to_mems.get(key_id, set())

    def _unlink_memory(self, memory_id: str) -> None:
        for kid in self._mem_to_keys.pop(memory_id, set()):
            mems = self._key_to_mems.get(kid, set())
            mems.discard(memory_id)
            if not mems:
                self._key_to_mems.pop(kid, None)

    @property
    def link_count(self) -> int:
        return sum(len(mids) for mids in self._key_to_mems.values())

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
            self._link(lnk["key_id"], lnk["memory_id"])
        # supersede 역방향 인덱스 빌드
        for mid, mem in self.memories.items():
            if mem.supersedes:
                self._superseded_by[mem.supersedes] = mid
        print(f"[graph] loaded {len(self.keys)} keys, {len(self.memories)} memories, {self.link_count} links")

    async def save(self) -> None:
        """비동기로 디스크에 저장하고 dirty 플래그 해제."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        links_list = [
            {"key_id": kid, "memory_id": mid}
            for kid, mids in self._key_to_mems.items()
            for mid in mids
        ]
        data = {
            "keys": {kid: _key_dict(k) for kid, k in self.keys.items()},
            "memories": {mid: _mem_dict(m) for mid, m in self.memories.items()},
            "links": links_list,
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        await asyncio.to_thread(GRAPH_FILE.write_text, content)
        self._dirty = False

    def mark_dirty(self) -> None:
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

        # concept 키는 배치 임베딩 유사도로 병합
        emb = await embed_text_async(concept)
        concept_keys = [(kid, key) for kid, key in self.keys.items() if key.key_type == "concept"]
        if concept_keys:
            ids = [kid for kid, _ in concept_keys]
            matrix = np.array([key.embedding for _, key in concept_keys])
            sims = _batch_cosine_sim(emb, matrix)
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= KEY_MERGE_THRESHOLD:
                return ids[best_idx]

        kid = _uid()
        self.keys[kid] = Key(id=kid, concept=concept, embedding=emb, key_type="concept")
        return kid

    def get_keys_for_memory(self, memory_id: str) -> list[str]:
        """메모리에 연결된 키 concept 목록."""
        return [
            self.keys[kid].concept
            for kid in self._mem_to_keys.get(memory_id, set())
            if kid in self.keys
        ]

    def _key_idf(self, key_id: str) -> float:
        """IDF: 많은 기억에 연결된 키일수록 가중치 낮춤."""
        freq = len(self._key_to_mems.get(key_id, set()))
        if freq <= 1:
            return 1.0
        idf = 1.0 / freq
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
                if not self._has_link(kid, mid):
                    self._link(kid, mid)
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
            old.depth = old.depth * 0.8 if old.depth >= DEPTH_DEEP_THRESHOLD else old.depth * 0.3

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
            self._superseded_by[old_id] = mid

            if key_concepts:
                key_concepts = _sanitize_keys(key_concepts)
                for concept in key_concepts:
                    kt = key_types.get(concept, "concept")
                    kid = await self.find_or_create_key(concept, key_type=kt)
                    self._link(kid, mid)
            else:
                # 기존 링크 복사 (스냅샷으로 순회 — 무한루프 방지)
                for kid in list(self._mem_to_keys.get(old_id, set())):
                    self._link(kid, mid)

            await self.save()
            return mid

    # ── N:M 검색 (멀티홉 + IDF) ──

    HOP_DECAY = 0.3
    TIME_HALF_LIFE = 30 * 24 * 3600  # 30일 반감기

    def _time_factor(self, mem: "Memory") -> float:
        """시간 decay. deep 기억은 거의 안 잊혀지고 shallow는 빠르게 사라짐."""
        age = time.time() - mem.created_at
        decay_rate = 1.0 - mem.depth * 0.7
        decay = math.exp(-age * decay_rate / self.TIME_HALF_LIFE)
        return 0.5 + 0.5 * decay

    async def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """두 경로로 검색: (A) 키 매칭 → 링크 → 기억, (B) content 직접 매칭."""
        if not self.memories:
            return []

        q_emb = await embed_text_async(query)  # lock 밖에서 API 호출

        async with self._lock:
            query_lower = query.lower().strip()
            mem_scores: dict[str, float] = {}
            mem_matched_keys: dict[str, list[str]] = {}
            mem_hop: dict[str, int] = {}

            # ── 경로 A: 키 배치 매칭 → 링크 → 기억 ──
            key_ids = list(self.keys.keys())
            key_sims: np.ndarray = np.array([])
            if key_ids:
                key_matrix = np.array([self.keys[kid].embedding for kid in key_ids])
                key_sims = _batch_cosine_sim(q_emb, key_matrix)

            key_scores: list[tuple[float, str]] = []
            for i, kid in enumerate(key_ids):
                key = self.keys[kid]
                if key.key_type in ("name", "proper_noun"):
                    if key.concept.lower() in query_lower:
                        key_scores.append((1.0, kid))
                elif float(key_sims[i]) >= 0.35:
                    key_scores.append((float(key_sims[i]), kid))
            key_scores.sort(reverse=True)

            for key_sim, kid in key_scores[:10]:
                idf = self._key_idf(kid)
                for mem_id in self._key_to_mems.get(kid, set()):
                    if mem_id not in self.memories:
                        continue
                    mem = self.memories[mem_id]
                    depth_factor = 0.5 + mem.depth * 0.5
                    tf = self._time_factor(mem)
                    score = key_sim * idf * depth_factor * tf
                    mem_scores[mem_id] = mem_scores.get(mem_id, 0) + score
                    mem_matched_keys.setdefault(mem_id, []).append(self.keys[kid].concept)
                    mem_hop[mem_id] = 1

            # ── 경로 B: content 배치 직접 매칭 ──
            mem_ids = list(self.memories.keys())
            if mem_ids:
                mem_matrix = np.array([self.memories[mid].embedding for mid in mem_ids])
                content_sims = _batch_cosine_sim(q_emb, mem_matrix)
                for i, mid in enumerate(mem_ids):
                    c_sim = float(content_sims[i])
                    if c_sim >= 0.3:
                        mem = self.memories[mid]
                        depth_factor = 0.5 + mem.depth * 0.5
                        tf = self._time_factor(mem)
                        mem_scores[mid] = mem_scores.get(mid, 0) + c_sim * depth_factor * tf
                        mem_matched_keys.setdefault(mid, []).append("(content)")
                        if mid not in mem_hop:
                            mem_hop[mid] = 1

            # ── 2홉: 1홉 기억의 다른 키 → 새로운 기억 ──
            for mid in list(mem_scores.keys()):
                hop1_score = mem_scores[mid]
                for kid in self._mem_to_keys.get(mid, set()):
                    if kid not in self.keys:
                        continue
                    concept = self.keys[kid].concept
                    idf = self._key_idf(kid)
                    for other_mid in self._key_to_mems.get(kid, set()):
                        if other_mid == mid or other_mid not in self.memories:
                            continue
                        hop2_score = hop1_score * self.HOP_DECAY * idf
                        mem_scores[other_mid] = mem_scores.get(other_mid, 0) + hop2_score
                        mem_matched_keys.setdefault(other_mid, []).append(f"{concept}(via)")
                        if other_mid not in mem_hop:
                            mem_hop[other_mid] = 2

            # superseded 메모리(구버전)는 score에 패널티 적용
            for mid in list(mem_scores.keys()):
                if mid in self._superseded_by:
                    mem_scores[mid] *= 0.1

            ranked = sorted(mem_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            results = []
            for mid, score in ranked:
                mem = self.memories[mid]
                mem.depth = min(mem.depth + DEPTH_INCREMENT, DEPTH_MAX)
                mem.access_count += 1
                mem.last_accessed = time.time()
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
                    "superseded_by": self._superseded_by.get(mid),
                    "created_at": mem.created_at,
                })

            self.mark_dirty()

        await self.flush()  # lock 밖에서 I/O
        return results

    # ── 연관 기억 탐색 (공유 키 기반) ──

    def get_related(self, memory_id: str) -> list[dict]:
        """이 기억의 키를 공유하는 다른 기억 찾기 = 연상."""
        if memory_id not in self.memories:
            return []

        related: dict[str, dict] = {}
        for kid in self._mem_to_keys.get(memory_id, set()):
            concept = self.keys[kid].concept if kid in self.keys else "?"
            for mid in self._key_to_mems.get(kid, set()):
                if mid == memory_id or mid not in self.memories:
                    continue
                mem = self.memories[mid]
                if mid not in related:
                    related[mid] = {"id": mid, "content": mem.content, "shared_keys": [], "depth": round(mem.depth, 3)}
                if concept not in related[mid]["shared_keys"]:
                    related[mid]["shared_keys"].append(concept)

        return list(related.values())

    # ── 삭제 ──

    async def delete(self, memory_id: str) -> bool:
        async with self._lock:
            if memory_id not in self.memories:
                return False
            del self.memories[memory_id]
            self._unlink_memory(memory_id)
            # 고아 키 정리
            for kid in [k for k in list(self.keys) if k not in self._key_to_mems]:
                del self.keys[kid]
            # supersede 인덱스 정리
            self._superseded_by.pop(memory_id, None)
            for old_id, new_id in list(self._superseded_by.items()):
                if new_id == memory_id:
                    del self._superseded_by[old_id]
            await self.save()
            return True

    # ── 전체 조회 ──

    def list_all(self) -> list[dict]:
        return [
            {
                "id": mid,
                "content": mem.content,
                "keys": self.get_keys_for_memory(mid),
                "depth": round(mem.depth, 3),
                "access_count": mem.access_count,
                "supersedes": mem.supersedes,
                "created_at": mem.created_at,
            }
            for mid, mem in self.memories.items()
        ]


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
