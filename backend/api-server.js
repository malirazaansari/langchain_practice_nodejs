import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs/promises";
import dotenv from "dotenv";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { ChatOpenAI } from "@langchain/openai";
import { RetrievalQAChain } from "langchain/chains";
import { DynamicStructuredTool } from "@langchain/core/tools";
import {
  AgentExecutor,
  createOpenAIFunctionsAgent,
  initializeAgentExecutorWithOptions,
} from "langchain/agents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { getKnowledgeContext } from "./knowledgeBase.js";
import { z } from "zod";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
// pdfjsLib.GlobalWorkerOptions.workerSrc = undefined;
// import pdf from "pdf-parse";
// import pdf from "pdf-parse/lib/pdf-parse.mjs";

dotenv.config();

const app = express();
const PORT = 3002;

app.use((req, res, next) => {
  if (typeof res.setHeader !== "function") {
    res.setHeader = () => {};
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
const OPENAI_API_KEY =
  process.env.VITE_OPENAI_API_KEY || process.env.OPENAI_API_KEY;
const INDEX_NAME = process.env.PINECONE_INDEX || "rag-scanner";

// Ensure OPENAI_API_KEY is set in environment for LangChain
if (OPENAI_API_KEY && !process.env.OPENAI_API_KEY) {
  process.env.OPENAI_API_KEY = OPENAI_API_KEY;
  console.log("✅ Set OPENAI_API_KEY in process.env for LangChain");
}

let pinecone = null;
let index = null;
let openai = null;

async function initializeServices() {
  try {
    if (PINECONE_API_KEY) {
      pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
      index = pinecone.index(INDEX_NAME);
      console.log("✅ Pinecone initialized");
    }

    if (OPENAI_API_KEY) {
      openai = new OpenAI({ apiKey: OPENAI_API_KEY });
      console.log("✅ OpenAI initialized");
    }

    return true;
  } catch (error) {
    console.error("❌ Service initialization failed:", error);
    return false;
  }
}

// Extract device specifications from document text
// async function extractDeviceSpecs(textContent) {
//   if (!openai) {
//     console.warn("⚠️ OpenAI not available for device spec extraction");
//     return null;
//   }

//   try {
//     const prompt = `
// Analyze this hardware documentation and extract device specifications.

// IMPORTANT: This could be ANY type of hardware:
// - Servers/Motherboards (Dell, HP, Supermicro, ASUS, etc.)
// - Network switches (Arista, Cisco, Juniper, etc.)
// - GPUs (NVIDIA, AMD, Intel)
// - FPGAs (Xilinx, Intel/Altera)
// - Storage devices (SSDs, RAID controllers)
// - Other enterprise hardware

// Extract these fields (use "N/A" if not applicable):
// {
//   "vendor": "string (manufacturer name)",
//   "model": "string (full model name/number - REQUIRED)",
//   "socket": "string (CPU socket type OR interface type)",
//   "chipset": "string (chipset OR processor family)",
//   "cpuCount": "number (CPU count OR number of ports/channels)",
//   "ramSlots": "number (memory slots OR buffer size in GB)",
//   "firmwareVersion": "string (firmware/BIOS version)",
//   "deviceType": "string (server|switch|gpu|fpga|storage|network|other)"
// }

// Examples:
// - Network Switch: vendor="Arista", model="7050SX-64", socket="N/A", chipset="Broadcom Trident II", cpuCount=64 (ports), deviceType="switch"
// - FPGA: vendor="Xilinx", model="Zynq UltraScale+", socket="N/A", chipset="ARM Cortex-A53", cpuCount=4 (cores), deviceType="fpga"
// - Server: vendor="Dell", model="PowerEdge R750", socket="LGA4189", chipset="Intel C621A", cpuCount=2, deviceType="server"

// Document content:
// ${textContent.slice(0, 6000)}

// Return ONLY valid JSON with ALL fields filled (use "N/A" for non-applicable fields, not empty strings):`;

//     const response = await openai.chat.completions.create({
//       model: "gpt-4o-mini",
//       messages: [{ role: "user", content: prompt }],
//       temperature: 0,
//       max_tokens: 500,
//     });

//     let content = response.choices[0].message.content || "{}";
//     content = content
//       .replace(/```json\s*/g, "")
//       .replace(/```\s*/g, "")
//       .trim();

//     const result = JSON.parse(content);

//     // Validate that we got meaningful data (not all empty strings)
//     const hasValidData = result.vendor || result.model || result.deviceType;

//     if (!hasValidData) {
//       console.warn(
//         "⚠️ AI returned empty specs, likely unable to parse document"
//       );
//       return null;
//     }

//     console.log("🤖 AI extracted device specs:", result);
//     return result;
//   } catch (error) {
//     console.error("⚠️ Device spec extraction failed:", error);
//     return null;
//   }
// }
async function extractDeviceSpecs(inputContent) {
  if (!openai) {
    console.warn("⚠️ OpenAI not available for device spec extraction");
    return null;
  }

  let textContent = "";

  try {
    // 🧩 Detect if input is a Buffer (PDF file)
    if (Buffer.isBuffer(inputContent)) {
      console.log("📄 Detected PDF buffer, extracting text via pdfjs-dist...");

      const uint8Array = new Uint8Array(inputContent);
      const pdf = await pdfjsLib.getDocument({ data: uint8Array }).promise;
      // const pdf = await pdfjsLib.getDocument({ data: inputContent }).promise;
      let fullText = "";

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const text = await page.getTextContent();
        const pageText = text.items.map((t) => t.str).join(" ");
        fullText += pageText + "\n";
      }

      textContent = fullText;
      console.log(`📖 Extracted ${textContent.length} readable characters`);
    } else if (typeof inputContent === "string") {
      // Text already provided (for future flexibility)
      textContent = inputContent;
      console.log(`📖 Received plain text input (${textContent.length} chars)`);
    } else {
      throw new Error("Invalid input: must be a text string or PDF buffer");
    }

    // 🧠 Build AI extraction prompt
    const prompt = `
Analyze this hardware documentation and extract device specifications.

IMPORTANT: This could be ANY type of hardware:
- Servers/Motherboards (Dell, HP, Supermicro, ASUS, etc.)
- Network switches (Arista, Cisco, Juniper, etc.)
- GPUs (NVIDIA, AMD, Intel)
- FPGAs (Xilinx, Intel/Altera)
- Storage devices (SSDs, RAID controllers)
- Other enterprise hardware

Extract these fields (use "N/A" if not applicable):
{
  "vendor": "string (manufacturer name)",
  "model": "string (full model name/number - REQUIRED)",
  "socket": "string (CPU socket type OR interface type)",
  "chipset": "string (chipset OR processor family)",
  "cpuCount": "number (CPU count OR number of ports/channels)",
  "ramSlots": "number (memory slots OR buffer size in GB)",
  "firmwareVersion": "string (firmware/BIOS version)",
  "deviceType": "string (server|switch|gpu|fpga|storage|network|other)"
}

Examples:
- Network Switch: vendor="Arista", model="7050SX-64", socket="N/A", chipset="Broadcom Trident II", cpuCount=64 (ports), deviceType="switch"
- FPGA: vendor="Xilinx", model="Zynq UltraScale+", socket="N/A", chipset="ARM Cortex-A53", cpuCount=4 (cores), deviceType="fpga"
- Server: vendor="Dell", model="PowerEdge R750", socket="LGA4189", chipset="Intel C621A", cpuCount=2, deviceType="server"

Document content:
${textContent.slice(0, 6000)}

Return ONLY valid JSON with ALL fields filled (use "N/A" for non-applicable fields, not empty strings):`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: 500,
    });

    let content = response.choices[0].message.content || "{}";
    content = content
      .replace(/```json\s*/g, "")
      .replace(/```\s*/g, "")
      .trim();

    const result = JSON.parse(content);

    // ✅ Ensure at least one key is meaningful
    const hasValidData =
      result.vendor !== "N/A" ||
      result.model !== "N/A" ||
      result.deviceType !== "N/A";
    if (!hasValidData) {
      console.warn(
        "⚠️ AI returned empty or generic specs — check document readability"
      );
      return null;
    }

    console.log("🤖 AI extracted device specs:", result);
    return result;
  } catch (error) {
    console.error("❌ Device spec extraction failed:", error);
    return null;
  }
}

// Health check endpoint
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    services: {
      pinecone: !!pinecone,
      openai: !!openai,
    },
    timestamp: new Date().toISOString(),
  });
});

// ULTRA-FAST endpoint: Only extract device specs (no storage, no chunking)
app.post(
  "/documents/extract-specs-only",
  upload.single("document"),
  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "No file provided",
      });
    }

    const file = req.file;
    const startTime = Date.now();

    console.log(
      `⚡ ULTRA-FAST spec extraction: ${file.originalname} (${(file.size / 1024).toFixed(1)}KB)`
    );

    if (!openai) {
      return res.status(503).json({
        success: false,
        error: "OpenAI not available",
      });
    }

    try {
      const textContent = file.buffer.toString("utf8");
      console.log(`📖 Text extracted (${textContent.length} chars)`);

      // Direct AI spec extraction
      // const extractedSpecs = await extractDeviceSpecs(textContent);
      const extractedSpecs = await extractDeviceSpecs(file.buffer);

      const totalTime = Date.now() - startTime;
      console.log(`⚡ ULTRA-FAST extraction complete in ${totalTime}ms`);

      res.json({
        success: true,
        message: `Specs extracted in ${totalTime}ms`,
        extractedSpecs,
        filename: file.originalname,
        processingTimeMs: totalTime,
        mode: "ultra-fast-specs-only",
        note: "No database storage - maximum speed analysis",
      });
    } catch (error) {
      console.error("❌ Ultra-fast spec extraction error:", error);
      res.status(500).json({
        success: false,
        error: "Spec extraction failed",
        message: error.message,
      });
    }
  }
);

// Document search endpoint - searches ALL documents in Pinecone DB
app.post("/documents/search", async (req, res) => {
  const {
    query,
    topK = 5,
    minRelevance = 0.3,
    model = "gpt-4o-mini",
  } = req.body;

  console.log(
    `🔍 Search request: "${query}" (topK: ${topK}, minRelevance: ${minRelevance})`
  );
  console.log(`🧩 Using model for synthesis: ${model}`);

  if (!index || !openai) {
    return res.status(503).json({
      error: "Services not available",
      message: "Pinecone or OpenAI not initialized",
    });
  }

  try {
    // Generate query embedding
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: query,
      dimensions: 512,
    });

    const queryEmbedding = embeddingResponse.data[0].embedding;

    // Search ALL documents in Pinecone (no filters - searches entire DB)
    const searchResults = await index.query({
      vector: queryEmbedding,
      topK: parseInt(topK),
      includeMetadata: true,
      // No filter - searches ALL vectors in the entire database
    });

    // Filter by relevance threshold and categorize results
    const allMatches =
      searchResults.matches?.map((match) => ({
        id: match.id,
        content: match.metadata?.content || "",
        source: match.metadata?.source || "Unknown",
        relevance: match.score || 0,
        category: match.metadata?.category || "hardware",
        type: match.metadata?.type || "unknown",
        metadata: match.metadata || {},
      })) || [];

    // Filter by minimum relevance
    const relevantMatches = allMatches.filter(
      (match) => match.relevance >= minRelevance
    );

    // Categorize results by document type
    const resultsByType = {
      hardware_docs: relevantMatches.filter(
        (r) => r.type === "batch_upload" || r.source.includes(".md")
      ),
      user_uploads: relevantMatches.filter(
        (r) => r.type === "api_uploaded_document"
      ),
      all_results: relevantMatches,
    };

    console.log(
      `✅ Found ${allMatches.length} total matches, ${relevantMatches.length} above relevance threshold`
    );
    console.log(
      `📚 Hardware docs: ${resultsByType.hardware_docs.length}, User uploads: ${resultsByType.user_uploads.length}`
    );

    res.json({
      success: true,
      query,
      results: relevantMatches,
      resultsByType,
      count: relevantMatches.length,
      totalMatches: allMatches.length,
      hasRelevantResults: relevantMatches.length > 0,
      searchedEntireDB: true,
    });
  } catch (error) {
    console.error("❌ Search error:", error);
    res.status(500).json({
      success: false,
      error: "Search failed",
      message: error.message,
    });
  }
});

app.post("/documents/answer", async (req, res) => {
  const {
    query,
    model = "gpt-4o-mini",
    conversationHistory = [],
  } = req.body || {};
  console.log(`🧠 [Agent] Query: "${query}"`);

  if (!query || query.trim().length === 0) {
    return res.status(400).json({ success: false, error: "Missing query" });
  }

  try {
    // Initialize LLM with function calling support
    const llm = new ChatOpenAI({
      modelName: model,
      temperature: 0.5, // Increased for more creative responses
      openAIApiKey: OPENAI_API_KEY,
    });

    // Define knowledge base search tool
    const knowledgeBaseTool = new DynamicStructuredTool({
      name: "search_hardware_docs",
      description:
        "Search the hardware documentation database for SPECIFIC technical details you don't already know. Use this when:\n" +
        "- User asks about specific hardware models or part numbers\n" +
        "- You need exact specifications or compatibility information\n" +
        "- Question involves vendor-specific documentation\n" +
        "- You need detailed troubleshooting steps from manuals\n" +
        "DO NOT use for common commands or general knowledge you already have.",
      schema: z.object({
        query: z
          .string()
          .describe("The search query to find relevant hardware documentation"),
        topK: z
          .number()
          .optional()
          .default(5)
          .describe("Number of results to return (default 5)"),
      }),
      func: async ({ query, topK = 5 }) => {
        console.log(
          `🔍 [Tool Called] search_hardware_docs with query: "${query}"`
        );

        try {
          const kb = await getKnowledgeContext(query, topK);

          if (!kb.success || kb.documents.length === 0) {
            console.warn("⚠️ [Tool] No docs found");
            return "No relevant documentation found in the knowledge base.";
          }

          console.log(
            `✅ [Tool Result] Found ${kb.documents.length} relevant documents`
          );
          return kb.context; // Return formatted context string
        } catch (error) {
          console.error("❌ [Tool Error]:", error);
          return `Error searching documentation: ${error.message}`;
        }
      },
    });

    // Create agent prompt with system instructions
    const agentPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are Basil, an expert hardware diagnostics assistant with deep knowledge of servers, GPUs, CPUs, networking, and system administration.

HOW TO RESPOND:
1. Think about the user's question first
2. If you know the answer from your expertise, respond directly with confidence
3. If you need specific documentation or hardware specs you don't know, THEN use the search_hardware_docs tool
4. After getting tool results, synthesize the information and provide a helpful answer
5. ALWAYS include a relevant command when applicable (wrapped in backticks)

RESPONSE FORMAT:
- Provide clear, concise explanations (2-3 sentences)
- Include bash/shell commands in backticks: \`command here\`
- Be conversational and helpful
- Never return empty responses - always provide an answer

EXAMPLES:
User: "How do I check NVIDIA GPU temperature?"
You: "To monitor NVIDIA GPU temperature, use \`nvidia-smi\` which displays real-time temperature, utilization, and memory usage. You can also use \`watch -n 1 nvidia-smi\` for continuous monitoring."

User: "What's the best way to diagnose memory issues?"
You: "Start with \`free -h\` to check overall memory usage, then use \`vmstat 1\` to monitor memory statistics in real-time. For detailed per-process analysis, run \`top\` or \`htop\`."

Current conversation context: {context}`,
      ],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    // Build conversation context from history
    const contextStr =
      conversationHistory.length > 0
        ? conversationHistory
            .slice(-6)
            .map(
              (msg, i) =>
                `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`
            )
            .join("\n")
        : "No previous context";

    // Create agent with tools
    const agent = await createOpenAIFunctionsAgent({
      llm,
      tools: [knowledgeBaseTool],
      prompt: agentPrompt,
    });

    console.log("🤖 [Agent] Starting execution...");
    const agentExecutor = new AgentExecutor({
      agent,
      tools: [knowledgeBaseTool],
      verbose: true,
      maxIterations: 5,
    });
    let result;
    try {
      // const agentExecutor = await initializeAgentExecutorWithOptions(
      //   [knowledgeBaseTool],
      //   llm,
      //   {
      //     agentType: "openai-functions", // ensures tool->LLM->final answer loop
      //     verbose: true,
      //     maxIterations: 5,
      //   }
      // );
      // // Execute agent call
      // result = await agentExecutor.call({
      //   input: query,
      //   context: contextStr,
      // });
      console.log("🤖 [Agent] Starting execution...");
      result = await agentExecutor.invoke({
        input: query,
        context: contextStr,
      });
    } catch (invokeError) {
      console.error("❌ [Agent] Invocation error:", invokeError.message);
      console.error("Stack:", invokeError.stack);

      // Fallback to direct LLM response if agent fails completely
      console.log("🔄 [Agent] Falling back to direct LLM...");
      const fallbackResponse = await llm.invoke([
        {
          role: "system",
          content:
            "You are Basil, a helpful hardware diagnostics assistant. Provide clear, concise answers with relevant commands.",
        },
        { role: "user", content: query },
      ]);

      result = {
        output:
          fallbackResponse.content ||
          "I apologize, but I encountered an error processing your request.",
        intermediateSteps: [],
      };
    }

    console.log("✅ [Agent] Execution complete");
    console.log("📤 [Agent] Raw output:", result.output);
    console.log("📤 [Agent] Output type:", typeof result.output);
    console.log("📤 [Agent] Output length:", result.output?.length);
    console.log(
      "📊 [Agent] Intermediate steps:",
      result.intermediateSteps?.length || 0
    );

    // Check if output is valid
    if (!result.output || result.output.trim().length === 0) {
      console.error(
        "⚠️ [Agent] No output generated - returning generic message"
      );
      return res.json({
        success: true,
        source: "langchain-agent",
        answer:
          "I apologize, but I couldn't generate a response. The knowledge base might not be available. However, I can help with general hardware diagnostics questions.",
        assistant_response: {
          message:
            "I apologize, but I couldn't generate a response. The knowledge base might not be available. However, I can help with general hardware diagnostics questions.",
          command_text: "N/A",
          command: "N/A",
          confidence: 0.3,
          sources: [],
        },
        toolCalls: 0,
      });
    }

    // Extract command from agent's response (natural language parsing)
    let extractedCommand = "";
    let commandText = "Check GPU Temperature";

    // Look for code blocks (triple backticks or single backticks)
    // First try triple backticks (```bash\ncommand\n```)
    let codeBlockMatch = result.output.match(
      /```(?:bash|sh)?\s*\n([^\n]+)\n```/
    );

    if (!codeBlockMatch) {
      // Try single backticks (`command`)
      codeBlockMatch = result.output.match(/`([^`]+)`/);
    }

    if (codeBlockMatch) {
      extractedCommand = codeBlockMatch[1].trim();

      // Extract a title for the command from the text before it
      const beforeCommand = result.output.substring(
        0,
        result.output.indexOf(codeBlockMatch[0])
      );
      const titleMatch = beforeCommand.match(
        /(?:check|using?|run|execute|command|display|monitor|show)[\s:]+([^.!?\n]{5,50})/i
      );
      if (titleMatch) {
        commandText = titleMatch[1].trim().replace(/^the\s+/i, "");
      }

      console.log(`✅ [Agent] Extracted command: "${extractedCommand}"`);
      console.log(`✅ [Agent] Command title: "${commandText}"`);
    } else {
      console.log(`⚠️ [Agent] No command found in response`);
    }

    // Build structured response from agent output
    const assistant_response = {
      message: result.output,
      command_text: commandText,
      command: extractedCommand || "N/A",
      confidence: result.intermediateSteps?.length > 0 ? 0.9 : 0.8,
      sources:
        result.intermediateSteps?.length > 0
          ? ["Hardware Documentation KB"]
          : [],
    };

    return res.json({
      success: true,
      source: "langchain-agent",
      answer: assistant_response.message,
      assistant_response,
      toolCalls: result.intermediateSteps?.length || 0,
    });
  } catch (err) {
    console.error("❌ [Agent Error]:", err);
    res.status(500).json({
      success: false,
      error: "Agent execution failed",
      message: err.message,
    });
  }
});

// Document count endpoint
app.get("/documents/count", async (req, res) => {
  if (!index) {
    return res.status(503).json({
      error: "Pinecone not available",
      count: 0,
    });
  }

  try {
    const stats = await index.describeIndexStats();
    const count = stats.totalVectorCount || 0;

    console.log(`📊 Document count: ${count}`);

    res.json({
      success: true,
      count,
    });
  } catch (error) {
    console.error("❌ Count error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to get count",
      message: error.message,
      count: 0,
    });
  }
});

// OPTIMIZED: Fast device spec extraction endpoint
app.post(
  "/documents/analyze-for-setup",
  upload.single("document"),
  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "No file provided",
      });
    }

    const file = req.file;
    const { storeInDB = "false" } = req.body; // Parse as string from FormData
    const shouldStore = storeInDB === "true";

    const startTime = Date.now();
    console.log(
      `🔧 FAST device analysis: ${file.originalname} (${(file.size / 1024).toFixed(1)}KB) store: ${shouldStore}`
    );

    if (!openai) {
      return res.status(503).json({
        success: false,
        error: "OpenAI not available",
        message: "OpenAI not initialized",
      });
    }

    try {
      // Extract text content
      const textContent = file.buffer.toString("utf8");
      console.log(
        `📖 Extracted ${textContent.length} characters (${Date.now() - startTime}ms)`
      );

      // OPTIMIZATION 1: Extract device specs FIRST (fastest operation)
      const specsStartTime = Date.now();
      const extractedSpecs = await extractDeviceSpecs(file.buffer);
      // const extractedSpecs = await extractDeviceSpecs(textContent);
      console.log(`🤖 Specs extracted in ${Date.now() - specsStartTime}ms`);

      let chunksStored = 0;
      let processingTime = 0;

      // OPTIMIZATION 2: Only do expensive chunking/embedding if storing
      if (shouldStore && index) {
        console.log(`💾 Starting optimized Pinecone storage...`);
        const storageStartTime = Date.now();

        // OPTIMIZATION 3: Larger, more efficient chunks
        const chunkSize = 1200; // Increased from 800
        const overlap = 200; // Increased overlap for better context
        const chunks = [];

        // More efficient chunking
        for (let i = 0; i < textContent.length; i += chunkSize - overlap) {
          const chunk = textContent.slice(i, i + chunkSize).trim();
          if (chunk.length > 100) {
            // Higher minimum for quality
            chunks.push(chunk);
          }
        }

        console.log(`✂️ Created ${chunks.length} optimized chunks`);

        // OPTIMIZATION 4: Parallel embedding generation with larger batches
        const BATCH_SIZE = 10; // Increased from 3 to 10
        const batches = [];

        for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
          batches.push(chunks.slice(i, i + BATCH_SIZE));
        }

        // Process batches with parallel embedding calls
        for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
          const batch = batches[batchIndex];
          console.log(
            `🔢 Processing batch ${batchIndex + 1}/${batches.length} (${batch.length} chunks)`
          );

          // PARALLEL embedding generation
          const embeddingPromises = batch.map((chunk, localIndex) =>
            openai.embeddings
              .create({
                model: "text-embedding-3-small",
                input: chunk,
                dimensions: 512,
              })
              .then((response) => ({
                chunk,
                embedding: response.data[0].embedding,
                globalIndex: batchIndex * BATCH_SIZE + localIndex,
              }))
          );

          // Wait for all embeddings in this batch
          const embeddingResults = await Promise.all(embeddingPromises);

          // Prepare vectors for Pinecone
          const vectors = embeddingResults.map(
            ({ chunk, embedding, globalIndex }) => ({
              id: `setup-${file.originalname.replace(/[^a-zA-Z0-9]/g, "-")}-${globalIndex}-${Date.now()}`,
              values: embedding,
              metadata: {
                source: file.originalname,
                content: chunk,
                chunkIndex: globalIndex,
                totalChunks: chunks.length,
                uploadedAt: new Date().toISOString(),
                type: "user_device_document",
                category: "user_upload",
              },
            })
          );

          // OPTIMIZATION 5: Batch upsert to Pinecone
          await index.upsert(vectors);
          chunksStored += vectors.length;

          console.log(
            `📤 Batch ${batchIndex + 1}/${batches.length} uploaded (${vectors.length} vectors)`
          );
        }

        processingTime = Date.now() - storageStartTime;
        console.log(`💾 Pinecone storage completed in ${processingTime}ms`);
      }

      const totalTime = Date.now() - startTime;
      console.log(
        `✅ FAST analysis complete in ${totalTime}ms: specs=${!!extractedSpecs}, chunks=${chunksStored}`
      );

      res.json({
        success: true,
        message: `Device analyzed in ${totalTime}ms`,
        extractedSpecs,
        filename: file.originalname,
        chunksCreated: chunksStored,
        chunksStored,
        storedInDB: shouldStore && !!index,
        processingTimeMs: totalTime,
        storageTimeMs: processingTime,
        mode: "fast-analysis",
      });
    } catch (error) {
      console.error("❌ Device analysis error:", error);
      res.status(500).json({
        success: false,
        error: "Device analysis failed",
        message: error.message,
      });
    }
  }
);

// Document upload endpoint (general purpose storage)
app.post("/documents/upload", upload.single("document"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      error: "No file provided",
    });
  }

  const file = req.file;
  console.log(
    `📤 Upload request: ${file.originalname} (${(file.size / 1024 / 1024).toFixed(2)}MB)`
  );

  if (!index || !openai) {
    return res.status(503).json({
      success: false,
      error: "Services not available",
      message: "Pinecone or OpenAI not initialized",
    });
  }

  try {
    // Extract text content
    const textContent = file.buffer.toString("utf8");
    console.log(`📖 Extracted ${textContent.length} characters`);

    // Simple chunking
    const chunkSize = 800;
    const overlap = 150;
    const chunks = [];

    for (let i = 0; i < textContent.length; i += chunkSize - overlap) {
      const chunk = textContent.slice(i, i + chunkSize).trim();
      if (chunk.length > 50) {
        chunks.push(chunk);
      }
    }

    console.log(`✂️ Created ${chunks.length} chunks`);

    // Process in small batches to prevent memory issues
    let totalUploaded = 0;
    const batchSize = 3;

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batchChunks = chunks.slice(
        i,
        Math.min(i + batchSize, chunks.length)
      );
      const vectors = [];

      for (let j = 0; j < batchChunks.length; j++) {
        const chunkIndex = i + j;

        // Generate embedding
        const embeddingResponse = await openai.embeddings.create({
          model: "text-embedding-3-small",
          input: batchChunks[j],
          dimensions: 512,
        });

        vectors.push({
          id: `${file.originalname}-chunk-${chunkIndex}-${Date.now()}`,
          values: embeddingResponse.data[0].embedding,
          metadata: {
            source: file.originalname,
            content: batchChunks[j],
            chunkIndex: chunkIndex,
            totalChunks: chunks.length,
            uploadedAt: new Date().toISOString(),
            type: "api_uploaded_document",
          },
        });
      }

      // Upload batch to Pinecone
      await index.upsert(vectors);
      totalUploaded += vectors.length;

      console.log(
        `📤 Uploaded batch ${Math.floor(i / batchSize) + 1}: ${vectors.length} vectors`
      );
    }

    console.log(`✅ Upload complete: ${totalUploaded} chunks processed`);

    res.json({
      success: true,
      message: "Document processed successfully",
      chunksCreated: totalUploaded,
      filename: file.originalname,
    });
  } catch (error) {
    console.error("❌ Upload error:", error);
    res.status(500).json({
      success: false,
      error: "Upload processing failed",
      message: error.message,
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error("❌ Server error:", error);
  res.status(500).json({
    success: false,
    error: "Internal server error",
    message: error.message,
  });
});

// Start server
async function startServer() {
  const initialized = await initializeServices();

  if (!initialized) {
    console.warn(
      "⚠️ Some services failed to initialize, but server will start anyway"
    );
  }

  app.listen(PORT, () => {
    console.log(`\n🎯 RAG API Server running on http://localhost:${PORT}`);
    console.log(`📊 Available endpoints:
  GET  /health           - Service status
  POST /documents/search - Search documents  
  POST /documents/answer - LLM-first answer (model -> RAG fallback)
  GET  /documents/count  - Get document count
  POST /documents/upload - Upload new document
`);
    console.log("🔗 Your web app can now use API-based RAG!\n");
  });
}

startServer().catch(console.error);
