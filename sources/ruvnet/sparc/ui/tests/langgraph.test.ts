import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

// Initialize the model with API key from env
const model = new ChatAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  modelName: "claude-3-sonnet-20240229",
});

async function main() {
  // Create the prompt templates
  const haikuPrompt = ChatPromptTemplate.fromTemplate(
    "Write a haiku about programming. Only return the haiku, nothing else."
  );

  const analyzePrompt = ChatPromptTemplate.fromTemplate(
    "Analyze the theme of this haiku:\n{haiku}"
  );

  // Create the chain
  const chain = RunnableSequence.from([
    // First generate a haiku
    async () => {
      const haiku = await haikuPrompt
        .pipe(model)
        .pipe(new StringOutputParser())
        .invoke({});
      return { haiku };
    },
    // Then analyze it
    async (state: { haiku: string }) => {
      const analysis = await analyzePrompt
        .pipe(model)
        .pipe(new StringOutputParser())
        .invoke({ haiku: state.haiku });
      return {
        haiku: state.haiku,
        analysis,
      };
    },
  ]);

  try {
    console.log("Running LangChain test...\n");
    
    const result = await chain.invoke({});

    console.log("Generated Haiku:");
    console.log(result.haiku);
    console.log("\nAnalysis:");
    console.log(result.analysis);
    
    console.log("\nTest completed successfully!");
  } catch (error) {
    console.error("Test failed:", error);
  }
}

// Run the test
main().catch(console.error);
