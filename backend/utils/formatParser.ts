import { StructuredOutputParser } from "@langchain/core/output_parsers";

export const assistantOutputParser =
  StructuredOutputParser.fromNamesAndDescriptions({
    message: "2–3 sentence helpful reply for the user",
    confidence: "number between 0 and 1 showing how confident the model is",
    command_text: "short descriptive title for a suggested command (optional)",
    command: "exact CLI command to run (or 'N/A' if not applicable)",
  });

export async function safeParseLLMOutput(
  rawOutput: string | undefined,
  fallback: Record<string, any> = {}
): Promise<any> {
  try {
    if (!rawOutput) return fallback;

    const clean = rawOutput
      .replace(/```json|```/gi, "")
      .replace(/^json/i, "")
      .trim();

    try {
      return await assistantOutputParser.parse(clean);
    } catch {
      return JSON.parse(clean);
    }
  } catch (err) {
    console.warn("⚠️ [formatParser] Could not parse LLM output:", err);
    return {
      message: fallback.message || "I'm not sure, but here's what I found.",
      confidence: 0,
      command_text: fallback.command_text || "N/A",
      command: fallback.command || "N/A",
    };
  }
}

export default safeParseLLMOutput;
