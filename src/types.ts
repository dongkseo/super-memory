export interface Key {
  id: string;
  concept: string;
  embedding: number[];
  key_type: "concept" | "name" | "proper_noun";
}

export interface Memory {
  id: string;
  content: string;
  embedding: number[];
  created_at: number;
  source: Record<string, unknown> | null;
  supersedes: string | null;
  depth: number;
  access_count: number;
  last_accessed: number;
  namespace: string;
  ttl: number | null;
  links: string[];
}

export interface GraphData {
  keys: Record<string, Key>;
  memories: Record<string, Memory>;
  links: Array<{ key_id: string; memory_id: string }>;
}
