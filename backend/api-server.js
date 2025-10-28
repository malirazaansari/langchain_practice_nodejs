import express from "express";
import cors from "cors";
import multer from "multer";
import dotenv from "dotenv";
import OpenAI from "openai";
import path from "path";
import { Pinecone } from "@pinecone-database/pinecone";
import { ChatOpenAI } from "@langchain/openai";
import { DynamicStructuredTool } from "@langchain/core/tools";
import {
  AgentExecutor,
  createOpenAIFunctionsAgent,
} from "langchain/agents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { getKnowledgeContext } from "./knowledgeBase.js";
import { z } from "zod";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import YAML from "yaml";
import { StructuredOutputParser } from "langchain/output_parsers";

import {
  uploadDocumentsFromDir,
  uploadSingleFile,
} from "../uploads-doc-to-pinecone.mjs";

dotenv.config();

const app = express();
const PORT = 3002;

const isDev = process.env.NODE_ENV === "development";

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

// File-type aware text extraction helpers
function detectFileType(file) {
  const mime = (file?.mimetype || "").toLowerCase();
  const ext = (
    path.extname(file?.originalname || "").toLowerCase() || ""
  ).replace(".", "");

  // Prefer MIME when present
  if (mime.includes("pdf")) return "pdf";
  if (mime.includes("json")) return "json";
  if (mime.includes("yaml") || mime.includes("yml")) return "yaml";
  if (mime.startsWith("text/")) return "text";

  // Fallback to extension
  if (["pdf"].includes(ext)) return "pdf";
  if (["json"].includes(ext)) return "json";
  if (["yaml", "yml"].includes(ext)) return "yaml";
  if (["txt", "log", "md"].includes(ext)) return "text";

  return "unknown";
}

async function extractTextFromPDF(buffer) {
  try {
    console.log("📄 PDF detected, extracting text via pdfjs-dist...");
    const uint8Array = new Uint8Array(buffer);
    const pdf = await pdfjsLib.getDocument({ data: uint8Array }).promise;
    let fullText = "";
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const text = await page.getTextContent();
      const pageText = text.items.map((t) => t.str).join(" ");
      fullText += pageText + "\n";
    }
    console.log(`Extracted ${fullText.length} chars from PDF via pdfjs-dist`);
    return fullText;
  } catch (err) {
    console.warn(
      "⚠️ pdfjs-dist failed, falling back to pdf-parse:",
      err?.message
    );
    try {
      // Lazy-load pdf-parse to avoid startup crashes in certain environments
      let pdfParseFn;
      try {
        const mod = await import("pdf-parse");
        pdfParseFn = mod?.default || mod;
      } catch (importErr) {
        console.warn("⚠️ Unable to load pdf-parse module:", importErr?.message);
        // Give up on PDF fallback gracefully
        return "";
      }

      const parsed = await pdfParseFn(buffer);
      const text = parsed?.text || "";
      console.log(`Extracted ${text.length} chars from PDF via pdf-parse`);
      return text;
    } catch (err2) {
      console.error("❌ PDF extraction failed (both methods):", err2?.message);
      // Fall back to empty text instead of throwing to avoid crashing handlers
      return "";
    }
  }
}

async function extractTextFromUpload(file) {
  const type = detectFileType(file);
  try {
    switch (type) {
      case "pdf":
        return await extractTextFromPDF(file.buffer);
      case "json": {
        const raw = file.buffer.toString("utf8");
        try {
          const obj = JSON.parse(raw);
          return JSON.stringify(obj, null, 2);
        } catch {
          return raw; // malformed JSON; return as-is
        }
      }
      case "yaml": {
        const raw = file.buffer.toString("utf8");
        // Keep original text to preserve comments/formatting
        // Optional parse validation
        try {
          YAML.parse(raw);
        } catch {
          // Ignore parse errors; we'll still use raw content
        }
        return raw;
      }
      case "text":
      case "unknown":
      default:
        return file.buffer.toString("utf8");
    }
  } finally {
    // no-op but a convenient place for future resource cleanup
  }
}

async function extractDeviceSpecs(inputContent) {
  if (!openai) {
    console.warn("⚠️ OpenAI not available for device spec extraction");
    return null;
  }

  let textContent = "";

  try {
    if (typeof inputContent === "string") {
      textContent = inputContent;
      console.log(`📖 Received plain text input (${textContent.length} chars)`);
    } else {
      throw new Error(
        "Invalid input: extractDeviceSpecs expects a text string"
      );
    }

    if (!textContent || textContent.trim().length < 10) {
      console.error("Text content too short or empty - cannot extract specs");
      return null;
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
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content:
            "You are a hardware doc extractor. Return only a valid JSON object matching the schema. No extra text.",
        },
        { role: "user", content: prompt },
      ],
      temperature: 0,
      max_tokens: 500,
    });

    let raw = response.choices[0].message.content || "{}";
    let cleaned = raw
      .replace(/```json\s*/gi, "")
      .replace(/```\s*/g, "")
      .trim();

    let result;
    try {
      result = JSON.parse(cleaned);
    } catch (e) {
      // Try to salvage a JSON object by extracting the outermost braces
      const start = cleaned.indexOf("{");
      const end = cleaned.lastIndexOf("}");
      if (start !== -1 && end !== -1 && end > start) {
        const slice = cleaned.slice(start, end + 1);
        try {
          result = JSON.parse(slice);
        } catch (e2) {
          // Final attempt: ask the model to repair into valid JSON (JSON mode)
          const repairPrompt = `Fix the following into a valid JSON object matching this schema with all keys present; use \"N/A\" for unknown values:
{
  \"vendor\": \"string\",
  \"model\": \"string\",
  \"socket\": \"string\",
  \"chipset\": \"string\",
  \"cpuCount\": \"number\",
  \"ramSlots\": \"number\",
  \"firmwareVersion\": \"string\",
  \"deviceType\": \"string\"
}

Content:\n${cleaned}`;

          const repair = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            response_format: { type: "json_object" },
            messages: [
              { role: "system", content: "Return only a valid JSON object." },
              { role: "user", content: repairPrompt },
            ],
            temperature: 0,
            max_tokens: 400,
          });
          const repaired = (repair.choices[0].message.content || "{}").trim();
          result = JSON.parse(repaired);
        }
      } else {
        throw e;
      }
    }

    // ✅ Ensure at least one key is meaningful
    const hasValidData = result.vendor !== "N/A" || result.model !== "N/A";
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

const specsParser = StructuredOutputParser.fromZodSchema(
  z.object({
    title: z
      .string()
      .describe("Title of the hardware specification or summary"),
    sections: z
      .array(
        z.object({
          name: z
            .string()
            .describe("Section name like Processor, Memory, etc."),
          details: z
            .array(z.string())
            .describe("Bullet points for each section"),
        })
      )
      .describe("List of key specification sections"),
  })
);

// Parser for structured, citation-grounded assistant answers
const structuredAnswerParser = StructuredOutputParser.fromZodSchema(
  z.object({
    answer: z
      .string()
      .describe(
        "Concise answer strictly grounded in the retrieved documents with inline citations like [docId:page]."
      ),
    commands: z
      .array(
        z.object({
          command: z
            .string()
            .describe("Shell command to run; keep it single-purpose and safe."),
          risk: z
            .enum(["low", "medium", "high"])
            .describe("Operational risk level of running the command."),
          explanation: z
            .string()
            .describe("What the command does and why it's relevant."),
          consent: z
            .string()
            .describe(
              "Explicit consent text to show the user before execution, e.g., 'Proceed to run ... ?'"
            ),
        })
      )
      .default([])
      .describe(
        "Optional commands the user can run; include risk, explanation, and explicit consent text."
      ),
    citations: z
      .array(z.string())
      .default([])
      .describe(
        "List of citations used, e.g., [docA:1], matching the inline citations."
      ),
    confidence: z
      .number()
      .min(0)
      .max(1)
      .describe(
        "Assistant's confidence based on document coverage and clarity."
      ),
  })
);

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
      const fileType = detectFileType(file);
      const textContent = await extractTextFromUpload(file);
      console.log(
        `📖 Extracted ${textContent.length} characters from ${fileType.toUpperCase()}`
      );

      // Direct AI spec extraction from normalized text
      const extractedSpecs = await extractDeviceSpecs(textContent);

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

app.post("/documents/answer", async (req, res) => {
  const {
    query,
    model = "gpt-4o-mini",
    conversationHistory = [],
    appState: clientState = null,
  } = req.body || {};
  if (isDev) console.log(`🧠 [Agent] Query: "${query}"`);

  if (!query || query.trim().length === 0) {
    return res.status(400).json({ success: false, error: "Missing query" });
  }

  // Set headers for Server-Sent Events (SSE)
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no"); // Disable nginx buffering
  res.flushHeaders?.();

  // Helper to send SSE messages
  const sendSSE = (data) => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  sendSSE({
    type: "message_chunk",
    content: "Thinking...\n",
    fullMessage: "Thinking...\n",
  });

  try {
    // Derive a compact device/app context string from provided app state
    const state = clientState || {};
    const stateParts = [];
    try {
      const deviceLabel = state?.deviceType?.label || state?.deviceType?.id;
      const vendor = state?.vendor || state?.model?.vendor;
      const modelLabel = state?.model?.label || state?.model?.id;
      const socket = state?.model?.socket || state?.boardSpecs?.socket;
      const chipset = state?.model?.chipset || state?.boardSpecs?.chipset;
      const cpuCount = state?.boardSpecs?.cpuCount;
      const ramSlots = state?.boardSpecs?.ramSlots;
      const fw = state?.boardSpecs?.firmwareVersion;
      const bmcIp = state?.networkConfig?.bmcIp;
      const bmcUser = state?.networkConfig?.username;
      const bmcTested = state?.networkConfig?.tested;
      const env = state?.environment;

      if (deviceLabel) stateParts.push(`deviceType=${deviceLabel}`);
      if (vendor) stateParts.push(`vendor=${vendor}`);
      if (modelLabel) stateParts.push(`model=${modelLabel}`);
      if (socket) stateParts.push(`socket=${socket}`);
      if (chipset) stateParts.push(`chipset=${chipset}`);
      if (Number.isFinite(cpuCount)) stateParts.push(`cpuCount=${cpuCount}`);
      if (Number.isFinite(ramSlots)) stateParts.push(`ramSlots=${ramSlots}`);
      if (fw) stateParts.push(`firmware=${fw}`);
      if (bmcIp) stateParts.push(`bmcIp=${bmcIp}`);
      if (bmcUser) stateParts.push(`bmcUser=${bmcUser}`);
      if (typeof bmcTested === "boolean")
        stateParts.push(`bmcTested=${bmcTested}`);
      if (env) stateParts.push(`env=${env}`);
    } catch {}
    const stateStr = stateParts.length
      ? `Device/App State: ${stateParts.join(", ")}`
      : "Device/App State: none";

    // Initialize LLM with function calling support
    const llm = new ChatOpenAI({
      modelName: model,
      temperature: 0.2,
      openAIApiKey: OPENAI_API_KEY,
      streaming: true, // Enable streaming
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
        if (isDev)
          console.log(
            `🔍 [Tool Called] search_hardware_docs with query: "${query}"`
          );

        try {
          // Enrich search with vendor/model context when available
          sendSSE({ type: "status", content: "Searching documentation..." });
          const ctxVendor = state?.vendor || state?.model?.vendor || "";
          const ctxModel = state?.model?.label || state?.model?.id || "";
          const effectiveQuery = [ctxVendor, ctxModel, query]
            .filter(Boolean)
            .join(" ")
            .trim();

          const kb = await getKnowledgeContext(effectiveQuery || query, topK);

          if (!kb.success || kb.documents.length === 0) {
            console.warn("⚠️ [Tool] No docs found");
            return "No relevant documentation found in the knowledge base.";
          }

          if (isDev)
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
2. ALWAYS ground your answer on the Current device context provided below. If the user says "this device" or is ambiguous, assume they mean the device in the Current device context. If the chat history mentions a different device, prefer the Current device context.
3. If you need specific documentation or hardware specs you don't know, THEN use the search_hardware_docs tool (augmenting the query with vendor/model from context)
4. After getting tool results, synthesize the information and provide a helpful answer
5. ALWAYS include one concise, most-relevant command when applicable (wrapped in backticks). Avoid repeating the same command multiple times.
6. If the device context is missing critical details (e.g., vendor/model) and the question depends on them, ask ONE short clarifying question before giving a general best-effort answer.

RESPONSE FORMAT:
- Provide clear, concise explanations (2-3 sentences)
- Include bash/shell commands in backticks: \`command here\`
- When suggesting a command, append the risk level in brackets after the command: [RISK: low/medium/high]
- Be conversational and helpful
- Never return empty responses - always provide an answer

RISK LEVELS FOR COMMANDS:
- LOW: Read-only commands that cannot harm the system (ls, cat, free, top, nvidia-smi, lspci, dmesg, etc.)
- MEDIUM: Commands that modify configuration or restart services (systemctl restart, apt-get install, chmod, mkdir, etc.)
- HIGH: Destructive or dangerous commands (rm -rf, dd, mkfs, shutdown, reboot, chmod 777, kill -9, etc.)

EXAMPLES:
User: "How do I check NVIDIA GPU temperature?"
You: "To monitor NVIDIA GPU temperature, use \`nvidia-smi\` which displays real-time temperature, utilization, and memory usage. You can also use \`watch -n 1 nvidia-smi\` for continuous monitoring. [RISK: low]"

User: "What's the best way to diagnose memory issues?"
You: "Start with \`free -h\` to check overall memory usage, then use \`vmstat 1\` to monitor memory statistics in real-time. For detailed per-process analysis, run \`top\` or \`htop\`. [RISK: low]"

User: "How do I check if my server is overheating?"
You: "To monitor temperature sensors on your Dell PowerEdge R620, you can use the ipmitool command:

\`\`\`bash
ipmitool sdr type Temperature
\`\`\`
[RISK: low]

This displays temperature readings from all available sensors, helping you identify any overheating components."

Current device context (ground truth): {state}

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

    if (isDev) console.log("🤖 [Agent] Starting execution...");
    const agentExecutor = new AgentExecutor({
      agent,
      tools: [knowledgeBaseTool],
      verbose: false,
      maxIterations: 1,
    });

    let fullOutput = "";
    let streamedTokens = 0;

    try {
      // Use LangChain's native .stream() method for robust streaming
      if (isDev)
        console.log(
          "🤖 [Agent] Starting streaming execution with tools: search_hardware_docs"
        );

      const callbacks = [
        {
          handleToolStart(tool, input) {
            sendSSE({
              type: "status",
              content: `Searching documentation for: "${input.query}"...`,
            });
          },
          handleToolEnd(output) {
            sendSSE({
              type: "status",
              content: "Documentation found. Generating answer...",
            });
          },
          handleLLMNewToken(token) {
            fullOutput += token;
            streamedTokens++;
            sendSSE({
              type: "message_chunk",
              content: token,
              fullMessage: fullOutput,
            });
          },
          handleLLMEnd() {
            console.log(`✅ Stream completed (${streamedTokens} tokens)`);
          },
        },
      ];

      const stream = await agentExecutor.stream(
        {
          input: query,
          chat_history: conversationHistory,
          context: contextStr || "No previous context available",
          state: stateStr,
        },
        {
          callbacks,
        }
      );

      // Process the stream
      let result = { output: "", intermediateSteps: [] };
      for await (const chunk of stream) {
        // Agent executor streams different event types
        if (chunk.agent) {
          // Agent reasoning step
          console.log("🤔 [Agent] Reasoning:", chunk.agent);
        } else if (chunk.actions) {
          // Tool invocation
          console.log("🔧 [Agent] Invoking tools:", chunk.actions);
        } else if (chunk.steps) {
          // Tool results
          console.log("📊 [Agent] Tool results:", chunk.steps);
          result.intermediateSteps = chunk.steps;
        } else if (chunk.output) {
          // Final output
          result.output = chunk.output;
          console.log("✅ [Agent] Final output received");
        }
      }

      // If we didn't get output from stream, use fullOutput
      if (!result.output && fullOutput) {
        result.output = fullOutput;
      }

      if (isDev) console.log("✅ [Agent] Stream execution complete");
      if (isDev)
        console.log("📤 [Agent] Raw output:", result.output || fullOutput);

      let structuredSpecs = null;
      try {
        // Ask model again to structure if it looks like a hardware spec
        if (/spec/i.test(query) || /specification/i.test(result.output)) {
          const formatInstructions = specsParser.getFormatInstructions();

          if (isDev) console.log("🧩 [Parser] Structuring output...");
          const structuredPrompt = `
        Restructure the following hardware specification into JSON:
        ${formatInstructions}
        Content:
        ${result.output}
        `;

          const structuredResponse = await llm.invoke([
            {
              role: "system",
              content: "You are a JSON formatter for hardware specifications.",
            },
            { role: "user", content: structuredPrompt },
          ]);

          structuredSpecs = await specsParser.parse(structuredResponse.content);
          if (isDev)
            console.log(
              "✅ [Parser] Parsed structured specs:",
              structuredSpecs
            );
          if (structuredSpecs) {
            let formatted = `\n\n${structuredSpecs.title.toUpperCase()}\n`;
            structuredSpecs.sections.forEach((sec) => {
              formatted += `\n${sec.name}:\n`;
              sec.details.forEach((d) => {
                formatted += `• ${d}\n`;
              });
            });
            result.output += formatted;
          }
        }
      } catch (parseErr) {
        console.warn(
          "⚠️ [Parser] Failed to structure specs:",
          parseErr.message
        );
      }
      if (isDev) console.log("📤 [Agent] Output type:", typeof result.output);
      if (isDev)
        console.log("📤 [Agent] Output length:", result.output?.length);
      if (isDev)
        console.log(
          "📊 [Agent] Intermediate steps:",
          result.intermediateSteps?.length || 0
        );

      // Use streamed output or fallback to result.output
      const finalOutput = fullOutput || result.output || "";

      // Check if output is valid; if not, perform a robust fallback that still answers the user
      if (!finalOutput || finalOutput.trim().length === 0) {
        console.warn("⚠️ [Agent] Empty output. Performing RAG+LLM fallback...");

        // Try to retrieve some context directly from the knowledge base
        let kbContext = "";
        let kbSources = [];
        try {
          // const kb = await getKnowledgeContext(query, 5);

          const ctxVendor = state?.vendor || state?.model?.vendor || "";
          const ctxModel = state?.model?.label || state?.model?.id || "";
          const effectiveQuery = [ctxVendor, ctxModel, query]
            .filter(Boolean)
            .join(" ")
            .trim();

          const kbPromise = getKnowledgeContext(effectiveQuery || query, 5);
          sendSSE({ type: "status", content: "Retrieving docs from KB..." });
          const kb = await kbPromise;
          if (kb?.success && kb?.context) {
            kbContext = kb.context;
            kbSources =
              kb.documents?.map((d) => d.metadata?.source).filter(Boolean) ||
              [];
          }
        } catch (e) {
          console.warn(
            "⚠️ [Fallback] Knowledge base lookup failed:",
            e.message
          );
        }

        // Ask the model directly, optionally conditioning on retrieved context
        const directPrompt = kbContext
          ? `You're Basil, an expert hardware diagnostics assistant. Use the device/app state and the context below to answer the user's question in 2-3 sentences and include a relevant bash command in backticks when applicable.\n\n${stateStr}\n\nContext:\n${kbContext}\n\nQuestion: ${query}`
          : `You're Basil, an expert hardware diagnostics assistant. Use the device/app state to tailor your answer. Respond in 2-3 sentences and include a relevant bash command in backticks when applicable.\n\n${stateStr}\n\nQuestion: ${query}`;

        const direct = await llm.invoke(
          [
            {
              role: "system",
              content:
                "Be concise, accurate, and include commands when useful.",
            },
            { role: "user", content: directPrompt },
          ],
          {
            callbacks: [
              {
                handleLLMNewToken(token) {
                  fullOutput += token;
                  sendSSE({
                    type: "message_chunk",
                    content: token,
                    fullMessage: fullOutput,
                  });
                },
              },
            ],
          }
        );

        const outputText =
          typeof direct?.content === "string"
            ? direct.content
            : Array.isArray(direct?.content)
              ? direct.content
                  .map((c) => (typeof c === "string" ? c : c?.text))
                  .join("\n")
              : fullOutput || "";

        // Extract risk level from anywhere in the output
        let riskLevel = "medium";
        const riskMatch = outputText.match(/\[RISK:\s*(low|medium|high)\]/i);
        if (riskMatch) {
          riskLevel = riskMatch[1].toLowerCase();
        }

        // Extract a command if present
        let extractedCommand = "";
        let commandText = "";
        let codeBlockMatch = outputText.match(
          /```(?:bash|sh)?\s*\n([^\n]+)\n```/
        );
        if (!codeBlockMatch) codeBlockMatch = outputText.match(/`([^`]+)`/);
        if (codeBlockMatch) {
          extractedCommand = codeBlockMatch[1].trim();
          // Clean up any [RISK: ...] tags from inside the command
          extractedCommand = extractedCommand
            .replace(/\s*\[RISK:\s*(low|medium|high)\]\s*/gi, "")
            .trim();

          const beforeCommand = outputText.substring(
            0,
            outputText.indexOf(codeBlockMatch[0])
          );
          const titleMatch = beforeCommand.match(
            /(?:check|using?|run|execute|command|display|monitor|show)[\s:]+([^.!?\n]{5,50})/i
          );
          if (titleMatch)
            commandText = titleMatch[1].trim().replace(/^the\s+/i, "");
        }

        // Clean message by removing [RISK: ...] tags
        const cleanOutputText = outputText
          .replace(/\[RISK:\s*(low|medium|high)\]/gi, "")
          .trim();

        // Send final complete message via SSE
        sendSSE({
          type: "complete",
          data: {
            message: cleanOutputText || "I'm here to help with that device.",
            command_text: commandText || (extractedCommand ? "Command" : "N/A"),
            command: extractedCommand || "N/A",
            risk: riskLevel,
            confidence: 1,
            sources: kbSources,
          },
        });

        res.end();
        return;
      }

      // Extract risk level from anywhere in the output
      let riskLevel = "medium";
      const riskMatch = finalOutput.match(/\[RISK:\s*(low|medium|high)\]/i);
      if (riskMatch) {
        riskLevel = riskMatch[1].toLowerCase();
        if (isDev)
          console.log(`✅ [Agent] Extracted risk level: "${riskLevel}"`);
      } else {
        if (isDev)
          console.log(`⚠️ [Agent] No risk level found, defaulting to "medium"`);
      }

      let extractedCommand = "";
      let commandText = "";

      let codeBlockMatch = finalOutput.match(
        /```(?:bash|sh)?\s*\n([^\n]+)\n```/
      );

      if (!codeBlockMatch) {
        codeBlockMatch = finalOutput.match(/`([^`]+)`/);
      }

      if (codeBlockMatch) {
        extractedCommand = codeBlockMatch[1].trim();

        // Clean up any [RISK: ...] tags from inside the command
        extractedCommand = extractedCommand
          .replace(/\s*\[RISK:\s*(low|medium|high)\]\s*/gi, "")
          .trim();

        const beforeCommand = finalOutput.substring(
          0,
          finalOutput.indexOf(codeBlockMatch[0])
        );
        const titleMatch = beforeCommand.match(
          /(?:check|using?|run|execute|command|display|monitor|show)[\s:]+([^.!?\n]{5,50})/i
        );
        if (titleMatch) {
          commandText = titleMatch[1].trim().replace(/^the\s+/i, "");
        }

        if (isDev)
          console.log(`✅ [Agent] Extracted command: "${extractedCommand}"`);
        if (isDev) console.log(`✅ [Agent] Command title: "${commandText}"`);
      } else {
        if (isDev) console.log(`⚠️ [Agent] No command found in response`);
      }

      // Clean message by removing [RISK: ...] tags
      const cleanMessage = finalOutput
        .replace(/\[RISK:\s*(low|medium|high)\]/gi, "")
        .trim();

      // Send final complete message via SSE
      sendSSE({
        type: "complete",
        data: {
          message: cleanMessage,
          command_text: commandText || "N/A",
          command: extractedCommand || "N/A",
          risk: riskLevel,
          confidence: result.intermediateSteps?.length > 0 ? 0.9 : 0.8,
          sources:
            result.intermediateSteps?.length > 0
              ? ["Hardware Documentation KB"]
              : [],
          toolCalls: result.intermediateSteps?.length || 0,
        },
      });

      res.end();
    } catch (err) {
      console.error("❌ [Agent Error]:", err);
      sendSSE({
        type: "error",
        error: err.message,
      });
      res.end();
    }
  } catch (err) {
    console.error("❌ [Outer Error]:", err);
    sendSSE({
      type: "error",
      error: err.message,
    });
    res.end();
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
      const fileType = detectFileType(file);
      const textContent = await extractTextFromUpload(file);
      console.log(
        `📖 Extracted ${textContent.length} characters from ${fileType.toUpperCase()} (${Date.now() - startTime}ms)`
      );

      // OPTIMIZATION 1: Extract device specs FIRST (fastest operation)
      const specsStartTime = Date.now();
      const extractedSpecs = await extractDeviceSpecs(textContent);
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
    const textContent = await extractTextFromUpload(file);
    console.log(`📖 Extracted ${textContent.length} characters for chunking`);

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

app.post("/pinecone/upload", upload.single("document"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: "No file uploaded" });
  }

  try {
    const result = await uploadSingleFile(req.file);
    res.json({
      success: true,
      message: `Uploaded ${req.file.originalname}`,
      ...result,
    });
  } catch (error) {
    console.error("❌ Pinecone upload error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

app.post("/pinecone/upload-dir", async (req, res) => {
  try {
    const result = await uploadDocumentsFromDir("./docs");
    res.json({
      success: true,
      message: "All documents uploaded successfully",
      result,
    });
  } catch (error) {
    console.error("❌ Bulk Pinecone upload error:", error);
    res.status(500).json({ success: false, error: error.message });
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