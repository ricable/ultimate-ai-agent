import type { Message } from "@langchain/langgraph-sdk";

/**
 * Returns the content of a message as a string.
 *
 * @param content - The content of the message
 * @returns The content of the message as a string
 */
export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return content;
  const texts = content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text);
  return texts.join(" ");
}
