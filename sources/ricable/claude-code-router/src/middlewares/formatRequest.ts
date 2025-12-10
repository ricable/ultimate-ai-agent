import { Request, Response, NextFunction } from "express";
import { MessageCreateParamsBase } from "@anthropic-ai/sdk/resources/messages";
import OpenAI from "openai";
import { streamOpenAIResponse } from "../utils/stream";
import { log } from "../utils/log";

export const formatRequest = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  let {
    model,
    max_tokens,
    messages,
    system = [],
    temperature,
    metadata,
    tools,
    stream,
  }: MessageCreateParamsBase = req.body;
  log("formatRequest: ", req.body);
  try {
    // @ts-ignore
    const openAIMessages = Array.isArray(messages)
      ? messages.flatMap((anthropicMessage) => {
          const openAiMessagesFromThisAnthropicMessage = [];

          if (!Array.isArray(anthropicMessage.content)) {
            // Handle simple string content
            if (typeof anthropicMessage.content === "string") {
              openAiMessagesFromThisAnthropicMessage.push({
                role: anthropicMessage.role,
                content: anthropicMessage.content,
              });
            }
            // If content is not string and not array (e.g. null/undefined), it will result in an empty array, effectively skipping this message.
            return openAiMessagesFromThisAnthropicMessage;
          }

          // Handle array content
          if (anthropicMessage.role === "assistant") {
            const assistantMessage = {
              role: "assistant",
              content: null, // Will be populated if text parts exist
            };
            let textContent = "";
            // @ts-ignore
            const toolCalls = []; // Corrected type here

            anthropicMessage.content.forEach((contentPart) => {
              if (contentPart.type === "text") {
                textContent +=
                  (typeof contentPart.text === "string"
                    ? contentPart.text
                    : JSON.stringify(contentPart.text)) + "\\n";
              } else if (contentPart.type === "tool_use") {
                toolCalls.push({
                  id: contentPart.id,
                  type: "function",
                  function: {
                    name: contentPart.name,
                    arguments: JSON.stringify(contentPart.input),
                  },
                });
              }
            });

            const trimmedTextContent = textContent.trim();
            if (trimmedTextContent.length > 0) {
              // @ts-ignore
              assistantMessage.content = trimmedTextContent;
            }
            if (toolCalls.length > 0) {
              // @ts-ignore
              assistantMessage.tool_calls = toolCalls;
            }
            // @ts-ignore
            if (
              assistantMessage.content ||
              // @ts-ignore
              (assistantMessage.tool_calls &&
                // @ts-ignore
                assistantMessage.tool_calls.length > 0)
            ) {
              openAiMessagesFromThisAnthropicMessage.push(assistantMessage);
            }
          } else if (anthropicMessage.role === "user") {
            // For user messages, text parts are combined into one message.
            // Tool results are transformed into subsequent, separate 'tool' role messages.
            let userTextMessageContent = "";
            // @ts-ignore
            const subsequentToolMessages = [];

            anthropicMessage.content.forEach((contentPart) => {
              if (contentPart.type === "text") {
                userTextMessageContent +=
                  (typeof contentPart.text === "string"
                    ? contentPart.text
                    : JSON.stringify(contentPart.text)) + "\\n";
              } else if (contentPart.type === "tool_result") {
                // Each tool_result becomes a separate 'tool' message
                subsequentToolMessages.push({
                  role: "tool",
                  tool_call_id: contentPart.tool_use_id,
                  content:
                    typeof contentPart.content === "string"
                      ? contentPart.content
                      : JSON.stringify(contentPart.content),
                });
              }
            });

            const trimmedUserText = userTextMessageContent.trim();
            if (trimmedUserText.length > 0) {
              openAiMessagesFromThisAnthropicMessage.push({
                role: "user",
                content: trimmedUserText,
              });
            }
            // @ts-ignore
            openAiMessagesFromThisAnthropicMessage.push(
              // @ts-ignore
              ...subsequentToolMessages
            );
          } else {
            // Fallback for other roles (e.g. system, or custom roles if they were to appear here with array content)
            // This will combine all text parts into a single message for that role.
            let combinedContent = "";
            anthropicMessage.content.forEach((contentPart) => {
              if (contentPart.type === "text") {
                combinedContent +=
                  (typeof contentPart.text === "string"
                    ? contentPart.text
                    : JSON.stringify(contentPart.text)) + "\\n";
              } else {
                // For non-text parts in other roles, stringify them or handle as appropriate
                combinedContent += JSON.stringify(contentPart) + "\\n";
              }
            });
            const trimmedCombinedContent = combinedContent.trim();
            if (trimmedCombinedContent.length > 0) {
              openAiMessagesFromThisAnthropicMessage.push({
                role: anthropicMessage.role, // Cast needed as role could be other than 'user'/'assistant'
                content: trimmedCombinedContent,
              });
            }
          }
          return openAiMessagesFromThisAnthropicMessage;
        })
      : [];
    const systemMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] =
      Array.isArray(system)
        ? system.map((item) => ({
            role: "system",
            content: item.text,
          }))
        : [{ role: "system", content: system }];
    const data: any = {
      model,
      messages: [...systemMessages, ...openAIMessages],
      temperature,
      stream,
    };
    if (tools) {
      data.tools = tools
        .filter((tool) => !["StickerRequest"].includes(tool.name))
        .map((item: any) => ({
          type: "function",
          function: {
            name: item.name,
            description: item.description,
            parameters: item.input_schema,
          },
        }));
    }
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
    }
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    req.body = data;
    console.log(JSON.stringify(data.messages, null, 2));
  } catch (error) {
    console.error("Error in request processing:", error);
    const errorCompletion: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> =
      {
        async *[Symbol.asyncIterator]() {
          yield {
            id: `error_${Date.now()}`,
            created: Math.floor(Date.now() / 1000),
            model,
            object: "chat.completion.chunk",
            choices: [
              {
                index: 0,
                delta: {
                  content: `Error: ${(error as Error).message}`,
                },
                finish_reason: "stop",
              },
            ],
          };
        },
      };
    await streamOpenAIResponse(res, errorCompletion, model, req.body);
  }
  next();
};
