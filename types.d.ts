// Ambient module declarations to smooth TS migration for untyped/complex packages
declare module "@langchain/community/tools/tavily_search" {
  const TavilySearchResults: any;
  export { TavilySearchResults };
}

declare module "@pinecone-database/pinecone" {
  const Pinecone: any;
  export { Pinecone };
  export default Pinecone;
}

// Broad wildcard declarations for LangChain packages (refine later)
declare module "@langchain/*" {
  const whatever: any;
  export = whatever;
}

declare module "langchain/*" {
  const whatever: any;
  export = whatever;
}

declare module "cheerio" {
  const cheerio: any;
  export default cheerio;
}

declare module "pdfjs-dist/legacy/build/pdf.mjs" {
  const pdfjs: any;
  export = pdfjs;
}

declare module "openai" {
  const OpenAI: any;
  export default OpenAI;
}

declare module "pdf-parse" {
  const pdfparse: any;
  export default pdfparse;
}
