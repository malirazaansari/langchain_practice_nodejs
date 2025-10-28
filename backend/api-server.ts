import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { ChatOpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { DynamicStructuredTool } from "@langchain/core/tools";

import { getKnowledgeContext } from "./knowledgeBase";
import { z } from "zod";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";

import { uploadSingleFile, uploadDocumentsFromDir } from "../uploads-doc-to-pinecone";

dotenv.config();

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3002;

const isDev = process.env.NODE_ENV === "development";

app.use((req, res, next) => {
  if (typeof res.setHeader !== "function") {
    // lightweight shim when running in some test harnesses
    (res as any).setHeader = () => {};
  }
  next();
});

app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

app.use(express.json({ limit: "10mb" }));
const upload = multer({ storage: multer.memoryStorage() });

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const OPENAI_API_KEY = process.env.VITE_OPENAI_API_KEY || process.env.OPENAI_API_KEY;
const INDEX_NAME = process.env.PINECONE_INDEX || "rag-scanner";

if (OPENAI_API_KEY && !process.env.OPENAI_API_KEY) {
  process.env.OPENAI_API_KEY = OPENAI_API_KEY;
  console.log("✅ Set OPENAI_API_KEY in process.env for LangChain");
}

let pinecone: any = null;
let index: any = null;
let openai: any = null;

async function initializeServices() {
  try {
    if (PINECONE_API_KEY) {
      pinecone = new Pinecone({ apiKey: PINECONE_API_KEY } as any);
      index = pinecone.index(INDEX_NAME);
      console.log("✅ Pinecone initialized");
    }

    if (OPENAI_API_KEY) {
      openai = new OpenAI({ apiKey: OPENAI_API_KEY } as any);
      console.log("✅ OpenAI initialized");
    }

    return true;
  } catch (error) {
    console.error("❌ Service initialization failed:", error);
    return false;
  }
}

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", services: { pinecone: !!pinecone, openai: !!openai }, timestamp: new Date().toISOString() });
});

// ULTRA-FAST endpoint: extract specs only
app.post("/documents/extract-specs-only", upload.single("document"), async (req, res) => {
  if (!req.file) return res.status(400).json({ success: false, error: "No file provided" });
  const file = req.file;
  if (!openai) return res.status(503).json({ success: false, error: "OpenAI not available" });

  try {
    // simple text extraction using pdfjs or fallback handled elsewhere
    // For brevity, reuse uploads-doc-to-pinecone helper to process single file
    const result = await uploadSingleFile(file as any).catch(() => null);
    return res.json({ success: true, result });
  } catch (err: any) {
    console.error("❌ Ultra-fast spec extraction error:", err);
    return res.status(500).json({ success: false, error: err.message });
  }
});

// Document answer endpoint (simplified)
app.post("/documents/answer", async (req, res) => {
  const { query } = req.body || {};
  if (!query || query.trim().length === 0) return res.status(400).json({ success: false, error: "Missing query" });

  try {
    // Create a simple LLM and return a placeholder response for migration
    const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0.2 } as any) as any;
    const answer = `Simulated answer for query: ${query}`;
    res.json({ success: true, answer });
  } catch (err: any) {
    console.error("❌ /documents/answer error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.post("/pinecone/upload", upload.single("document"), async (req, res) => {
  if (!req.file) return res.status(400).json({ success: false, error: "No file uploaded" });
  try {
    const result = await uploadSingleFile(req.file as any);
    res.json({ success: true, message: `Uploaded ${req.file.originalname}`, ...result });
  } catch (error: any) {
    console.error("❌ Pinecone upload error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post("/pinecone/upload-dir", async (req, res) => {
  try {
    const result = await uploadDocumentsFromDir("./docs");
    res.json({ success: true, message: "All documents uploaded successfully", result });
  } catch (error: any) {
    console.error("❌ Bulk Pinecone upload error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Error middleware
app.use((error: any, req: any, res: any, next: any) => {
  console.error("❌ Server error:", error);
  res.status(500).json({ success: false, error: "Internal server error", message: error.message });
});

export async function startServer() {
  const initialized = await initializeServices();
  if (!initialized) console.warn("⚠️ Some services failed to initialize, but server will start anyway");

  app.listen(PORT, () => {
    console.log(`\n🎯 RAG API Server running on http://localhost:${PORT}`);
  });
}

if (require.main === module) {
  startServer().catch(console.error);
}

export default app;
