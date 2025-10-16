import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { index } from "./pineconeClient.js";

export async function getKnowledgeContext(query, topK = 5, minRelevance = 0.3) {
  console.log(`📚 [KnowledgeBase] Retrieving context for: "${query}"`);

  // Check if Pinecone is available
  if (!index) {
    console.warn(
      "⚠️ [KnowledgeBase] Pinecone not initialized - returning empty context"
    );
    return {
      success: false,
      context:
        "Pinecone database is not available. Please configure PINECONE_API_KEY in .env file.",
      documents: [],
    };
  }

  try {
    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-small",
      dimensions: 512,
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex: index,
    });

    const results = await vectorStore.similaritySearch(query, topK);

    if (!results || results.length === 0) {
      console.log("❌ [KnowledgeBase] No relevant docs found");
      return { success: false, context: "", documents: [] };
    }

    const context = results
      .map((doc, i) => {
        const meta = doc.metadata || {};
        return `${i + 1}. Source: ${meta.source || "Unknown"}\n${doc.pageContent.slice(0, 800)}`;
      })
      .join("\n\n");

    console.log(`✅ [KnowledgeBase] Found ${results.length} relevant docs`);
    return { success: true, context, documents: results };
  } catch (err) {
    console.error("❌ [KnowledgeBase] Retrieval error:", err);
    return { success: false, context: "", documents: [], error: err.message };
  }
}
