import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";

import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";

// Tool imports
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { createRetrieverTool } from "langchain/tools/retriever";

// Custom Data Source, Vector Stores
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

// Wrap runtime code in async main to avoid top-level await issues
async function main() {
  // Create Retriever
  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/docs/expression_language/"
  );
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();

  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  const retriever = vectorStore.asRetriever({ k: 2 });

  // Instantiate the model
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0.2,
  });

  // Prompt Template
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant."],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ] as any[]);

  // Tools
  const searchTool = new TavilySearchResults();
  const retrieverTool = createRetrieverTool(retriever as any, {
    name: "lcel_search",
    description:
      "Use this tool when searching for information about Lanchain Expression Language (LCEL)",
  });

  const tools = [searchTool, retrieverTool];

  const agent = await createOpenAIFunctionsAgent({
    llm: model as any,
    prompt: prompt as any,
    tools: tools as any,
  } as any);

  // Create the executor
  const agentExecutor = new (AgentExecutor as any)({
    agent: agent as any,
    tools: tools as any,
  } as any);

  // User Input

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const chat_history: Array<HumanMessage | AIMessage> = [];

  function askQuestion() {
    rl.question("User: ", async (input: string) => {
      if (input.toLowerCase() === "exit") {
        rl.close();
        return;
      }

      const response = await agentExecutor.invoke({
        input: input,
        chat_history: chat_history,
      });

      console.log("Agent: ", (response as any).output ?? response);

      chat_history.push(new HumanMessage(input));
      chat_history.push(new AIMessage((response as any).output ?? ""));

      askQuestion();
    });
  }

  askQuestion();
}

main().catch((e) => console.error(e));

export {};
