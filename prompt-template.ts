import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";
dotenv.config();

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.9,
});

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a talented chef.  Create a recipe based on a main ingredient provided by the user.",
  ],
  ["human", "{word}"],
]);

const chain = prompt.pipe(model as any);

async function main() {
  const response = await chain.invoke({
    word: "chicken",
  } as any);

  console.log(response);
}

main().catch((e) => console.error(e));

export {};
