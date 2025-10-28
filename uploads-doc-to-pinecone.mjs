import dotenv from "dotenv";
dotenv.config();

import fs from "fs/promises";
import path from "path";
import { JSDOM } from "jsdom";
import { Pinecone } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";

let pdf;
async function loadPdfModule() {
  if (!pdf) {
    const mod = await import("pdf-parse");
    pdf = mod.default || mod;
  }
  return pdf;
}

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const INDEX_NAME = process.env.PINECONE_INDEX || "rag-scanner";

// init clients
const embeddings = new OpenAIEmbeddings({
  apiKey: OPENAI_API_KEY,
  model: "text-embedding-3-small",
});

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pinecone.index(INDEX_NAME);

const CHUNK_SIZE = 800;

// 🧩 Text extractors
async function extractTextFromFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  try {
    if (ext === ".pdf") {
      const buffer = await fs.readFile(filePath);
      const pdfModule = await loadPdfModule();
      const { text } = await pdfModule(buffer);
      return text;
    } else if (ext === ".html" || ext === ".htm") {
      const html = await fs.readFile(filePath, "utf8");
      const dom = new JSDOM(html);
      return dom.window.document.body.textContent || "";
    } else if (
      [".txt", ".md", ".yml", ".yaml", ".json", ".csv"].includes(ext)
    ) {
      return await fs.readFile(filePath, "utf8");
    } else {
      console.log(`⚠️ Skipping unsupported file: ${filePath}`);
      return "";
    }
  } catch (err) {
    console.error(`❌ Error reading ${filePath}:`, err.message);
    return "";
  }
}

// 🧩 Split text
function chunkText(text, size = CHUNK_SIZE) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    const chunk = text.slice(i, i + size).trim();
    if (chunk.length > 50) chunks.push(chunk);
  }
  return chunks;
}

// 🧠 Skip already-uploaded files
async function alreadyUploaded(filename) {
  try {
    const result = await index.query({
      vector: Array(512).fill(0),
      filter: { source: filename },
      topK: 1,
      includeMetadata: true,
    });
    return result.matches?.length > 0;
  } catch (err) {
    console.warn(`⚠️ Could not check ${filename}: ${err.message}`);
    return false;
  }
}

// 📂 Recursively collect documents
async function findDocuments(dir) {
  const files = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...(await findDocuments(full)));
    else if (/\.(pdf|txt|md|yaml|yml|json|html|htm|csv)$/i.test(entry.name))
      files.push(full);
  }
  return files;
}

// 🚀 Main upload
async function uploadDocs() {
  console.log("🚀 Starting document ingestion...");

  const docsDir = path.resolve("./docs");
  const files = await findDocuments(docsDir);

  console.log(`📁 Found ${files.length} documents in ${docsDir}`);

  for (const file of files) {
    const filename = path.basename(file);
    if (await alreadyUploaded(filename)) {
      console.log(`⏭️ Skipping ${filename} (already in Pinecone)`);
      continue;
    }

    const text = await extractTextFromFile(file);
    if (!text || text.length < 100) {
      console.log(`⚠️ Skipping ${filename} (empty or too short)`);
      continue;
    }

    const chunks = chunkText(text);

    const docs = chunks.map((content, i) => ({
      pageContent: content,
      metadata: {
        source: filename,
        fullPath: file,
        chunkIndex: i,
      },
    }));

    console.log(
      `🧠 Creating embeddings for ${filename} (${chunks.length} chunks)`
    );

    await PineconeStore.fromDocuments(docs, embeddings, {
      pineconeIndex: index,
      //   namespace: "universal_uploads",
    });

    console.log(`✅ Uploaded ${filename} successfully`);
  }

  console.log("🎉 All documents processed and uploaded!");
}

uploadDocs().catch((err) => {
  console.error("❌ Ingestion error:", err);
});

// 🧩 Process a single document
async function processDocument(filePath) {
  const filename = path.basename(filePath);

  if (await alreadyUploaded(filename)) {
    console.log(`⏭️ Skipping ${filename} (already in Pinecone)`);
    return;
  }

  const text = await extractTextFromFile(filePath);
  if (!text || text.length < 100) {
    console.log(`⚠️ Skipping ${filename} (empty or too short)`);
    return;
  }

  const chunks = chunkText(text);

  const docs = chunks.map((content, i) => ({
    pageContent: content,
    metadata: {
      source: filename,
      fullPath: filePath,
      chunkIndex: i,
    },
  }));

  console.log(
    `🧠 Creating embeddings for ${filename} (${chunks.length} chunks)`
  );

  await PineconeStore.fromDocuments(docs, embeddings, {
    pineconeIndex: index,
  });

  console.log(`✅ Uploaded ${filename} successfully`);
}

export async function uploadSingleFile(file) {
  const tempPath = `/tmp/${file.originalname}`;
  await fs.writeFile(tempPath, file.buffer);
  await processDocument(tempPath);
  return { filename: file.originalname };
}

export async function uploadDocumentsFromDir(dirPath = "./docs") {
  const docs = await findDocuments(dirPath);
  for (const doc of docs) {
    await processDocument(doc);
  }
  return { total: docs.length };
}