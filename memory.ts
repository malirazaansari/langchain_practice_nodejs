import * as dotenv from "dotenv";
dotenv.config();

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { ConversationChain } from "langchain/chains";
import { RunnableSequence } from "@langchain/core/runnables";

import { BufferMemory } from "langchain/memory";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI assistant called Max. You are here to help answer questions and provide information to the best of your ability.
Chat History: {history}
{input}`);

const upstashMessageHistory = new UpstashRedisChatMessageHistory({
  sessionId: "mysession",
  config: {
    url: process.env.UPSTASH_REDIS_URL,
    token: process.env.UPSTASH_REST_TOKEN,
  },
}) as any;
const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashMessageHistory as any,
} as any);

const chain = RunnableSequence.from([
  {
    input: (initialInput: any) => initialInput.input,
    memory: () => memory.loadMemoryVariables({}),
  },
  {
    input: (previousOutput: any) => previousOutput.input,
    history: (previousOutput: any) => previousOutput.memory.history,
  },
  prompt as any,
  model as any,
]);

async function main() {
  console.log("Updated Chat Memory", await memory.loadMemoryVariables({}));

  let inputs2 = {
    input: "What is the passphrase?",
  };

  const resp2 = await chain.invoke(inputs2 as any);
  console.log(resp2);
  await memory.saveContext(inputs2 as any, {
    output: (resp2 as any).content,
  });
}

main().catch((e) => console.error(e));

export {};
