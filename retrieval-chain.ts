import { ChatOpenAI } from "@langchain/openai";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

import * as dotenv from "dotenv";
dotenv.config();

async function main() {
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
  });

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the user's question from the following context: 
  {context}
  Question: {input}`
  );

  const chain = await createStuffDocumentsChain({
    llm: model as any,
    prompt: prompt as any,
  } as any);

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

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain as any,
    retriever: retriever as any,
  } as any);

  const response = await retrievalChain.invoke({
    input: "What is LCEL?",
  } as any);

  console.log(response);
}

main().catch((e) => console.error(e));

export {};
