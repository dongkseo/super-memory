import { config } from "dotenv";
config();

import OpenAI from "openai";

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const OPENAI_EMBEDDING_MODEL =
  process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small";
const LOCAL_EMBEDDING_MODEL =
  process.env.LOCAL_EMBEDDING_MODEL ?? "paraphrase-multilingual-MiniLM-L12-v2";

const _DEFAULT_BACKEND = OPENAI_API_KEY ? "openai" : "local";
export const EMBEDDING_BACKEND =
  process.env.EMBEDDING_BACKEND ?? _DEFAULT_BACKEND;

const EMBED_RETRIES = 3;

let _openaiClient: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  if (!_openaiClient) {
    _openaiClient = new OpenAI({ apiKey: OPENAI_API_KEY });
  }
  return _openaiClient;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _localModel: any = null;

async function getLocalModel() {
  if (!_localModel) {
    try {
      const { pipeline } = await import("@xenova/transformers");
      _localModel = await pipeline("feature-extraction", LOCAL_EMBEDDING_MODEL);
    } catch {
      throw new Error(
        "@xenova/transformers is not installed.\n" +
          "Install it with: npm install @xenova/transformers\n" +
          "Or set OPENAI_API_KEY to use OpenAI embeddings."
      );
    }
  }
  return _localModel;
}

async function embedLocal(text: string): Promise<number[]> {
  const model = await getLocalModel();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data[0]) as number[];
}

export async function embedTextAsync(text: string): Promise<number[]> {
  if (EMBEDDING_BACKEND === "local") {
    return embedLocal(text);
  }

  const client = getOpenAIClient();
  for (let attempt = 0; attempt < EMBED_RETRIES; attempt++) {
    try {
      const resp = await client.embeddings.create({
        model: OPENAI_EMBEDDING_MODEL,
        input: text,
      });
      return resp.data[0].embedding;
    } catch (err) {
      if (attempt === EMBED_RETRIES - 1) throw err;
      await new Promise((resolve) =>
        setTimeout(resolve, Math.pow(2, attempt) * 1000)
      );
    }
  }
  throw new Error("unreachable");
}
