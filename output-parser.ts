import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CommaSeparatedListOutputParser, StringOutputParser } from "@langchain/core/output_parsers";

import { z } from "zod";
import { StructuredOutputParser } from "langchain/output_parsers";
import * as dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.9,
});

async function callStringOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate("Tell a joke about {word}.");
  const outputParser = new StringOutputParser();

  const chain = (prompt.pipe(model as any) as any).pipe(outputParser);

  return await chain.invoke({ word: "dog" } as any);
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Provide 5 synonyms, seperated by commas, for a word that the user will provide.",
    ],
    ["human", "{word}"],
  ]);
  const outputParser = new CommaSeparatedListOutputParser();

  const chain = (prompt.pipe(model as any) as any).pipe(outputParser);

  return await chain.invoke({
    word: "happy",
  } as any);
}

async function callStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract information from the following phrase.\n{format_instructions}\n{phrase}"
  );

  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "name of the person",
    age: "age of person",
  });

  const chain = (prompt.pipe(model as any) as any).pipe(outputParser);

  return await chain.invoke({
    phrase: "Max is 30 years old",
    format_instructions: outputParser.getFormatInstructions(),
  } as any);
}

async function callZodStructuredParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    "Extract information from the following phrase.\n{format_instructions}\n{phrase}"
  );
  const zodSchema = z.object({
    recipe: z.string().describe("name of recipe"),
    ingredients: z.array(z.string()).describe("ingredients"),
  });

  // Cast to any to avoid cross-package/typing friction during migration
  const outputParser = StructuredOutputParser.fromZodSchema(zodSchema as any);

  const chain = (prompt.pipe(model as any) as any).pipe(outputParser);

  return await chain.invoke({
    phrase:
      "The ingredients for a Spaghetti Bolognese recipe are tomatoes, minced beef, garlic, wine and herbs.",
    format_instructions: outputParser.getFormatInstructions(),
  } as any);
}

async function main() {
  const response = await callZodStructuredParser();
  console.log(response);
}

main().catch((e) => console.error(e));

export {};
