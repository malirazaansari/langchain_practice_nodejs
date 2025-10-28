import { ChatOpenAI } from "@langchain/openai";
import readline from "readline";
import * as dotenv from "dotenv";
dotenv.config();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function chatCompletion(text: string) {
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.9,
  });

  const response = await model.invoke(text as any);

  // response may be an object depending on the LangChain API; keep same console behavior as JS file
  console.log("AI:", (response as any).content ?? response);
}

function getPrompt() {
  rl.question("Enter your prompt: ", (input: string) => {
    if (input.toUpperCase() === "EXIT") {
      rl.close();
    } else {
      chatCompletion(input).then(() => getPrompt());
    }
  });
}

getPrompt();

export {};
