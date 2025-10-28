import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const INDEX_NAME = process.env.PINECONE_INDEX || "rag-scanner";

let pinecone: any = null;
let index: any = null;

try {
  if (!PINECONE_API_KEY) {
    throw new Error("Missing PINECONE_API_KEY in .env");
  }

  pinecone = new Pinecone({ apiKey: PINECONE_API_KEY } as any);
  index = pinecone.index(INDEX_NAME);
  console.log(`✅ [Pinecone] Connected to index: ${INDEX_NAME}`);
} catch (err: any) {
  console.error("❌ [Pinecone] Initialization failed:", err.message);
}

export { pinecone, index };
