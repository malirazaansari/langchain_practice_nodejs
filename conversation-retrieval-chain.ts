import { ChatOpenAI } from "@langchain/openai";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

import * as dotenv from "dotenv";
dotenv.config();

async function main() {
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
  });

  const loader = new CheerioWebBaseLoader(
    "https://js.langchain.com/docs/expression_language/"
  );
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();

  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  const retriever = vectorstore.asRetriever({ k: 2 });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ] as any[]);

  const retrieverChain = await createHistoryAwareRetriever({
    llm: model as any,
    retriever: retriever as any,
    rephrasePrompt: retrieverPrompt as any,
  } as any);

  const chatHistory = [
    new HumanMessage("What does LCEL stand for?"),
    new AIMessage("LangChain Expression Language"),
  ];

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ] as any[]);

  const chain = await createStuffDocumentsChain({
    llm: model as any,
    prompt: prompt as any,
  } as any);

  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain as any,
    retriever: retrieverChain as any,
  } as any);

  const response = await conversationChain.invoke({
    chat_history: chatHistory,
    input: "What is it?",
  } as any);

  console.log(response);
}

main().catch((e) => console.error(e));

export {};
