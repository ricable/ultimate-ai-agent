import { Response } from "express";
import { OpenAI } from "openai";
import { log } from "./log";

interface ContentBlock {
  type: string;
  id?: string;
  name?: string;
  input?: any;
  text?: string;
}

interface MessageEvent {
  type: string;
  message?: {
    id: string;
    type: string;
    role: string;
    content: any[];
    model: string;
    stop_reason: string | null;
    stop_sequence: string | null;
    usage: {
      input_tokens: number;
      output_tokens: number;
    };
  };
  delta?: {
    stop_reason?: string;
    stop_sequence?: string | null;
    content?: ContentBlock[];
    type?: string;
    text?: string;
    partial_json?: string;
  };
  index?: number;
  content_block?: ContentBlock;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}

export async function streamOpenAIResponse(
  res: Response,
  completion: any,
  model: string,
  body: any
) {
  const write = (data: string) => {
    log("response: ", data);
    res.write(data);
  };
  const messageId = "msg_" + Date.now();
  if (!body.stream) {
    let content: any = [];
    if (completion.choices[0].message.content) {
      content = [ { text: completion.choices[0].message.content, type: "text" } ];
    }
    else if (completion.choices[0].message.tool_calls) {
      content = completion.choices[0].message.tool_calls.map((item: any) => {
        return {
          type: 'tool_use',
          id: item.id,
          name: item.function?.name,
          input: item.function?.arguments ? JSON.parse(item.function.arguments) : {},
        };
      });
    }

    const result = {
      id: messageId,
      type: "message",
      role: "assistant",
      // @ts-ignore
      content: content,
      stop_reason: completion.choices[0].finish_reason === 'tool_calls' ? "tool_use" : "end_turn",
      stop_sequence: null,
    };
    try {
      res.json(result);
      res.end();
      return;
    } catch (error) {
      log("Error sending response:", error);
      res.status(500).send("Internal Server Error");
    }
  }

  let contentBlockIndex = 0;
  let currentContentBlocks: ContentBlock[] = [];

  // Send message_start event
  const messageStart: MessageEvent = {
    type: "message_start",
    message: {
      id: messageId,
      type: "message",
      role: "assistant",
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 1, output_tokens: 1 },
    },
  };
  write(`event: message_start\ndata: ${JSON.stringify(messageStart)}\n\n`);

  let isToolUse = false;
  let toolUseJson = "";
  let hasStartedTextBlock = false;
  let currentToolCallId: string | null = null;
  let toolCallJsonMap = new Map<string, string>();

  try {
    for await (const chunk of completion) {
      log("Processing chunk:", chunk);
      const delta = chunk.choices[0].delta;

      if (delta.tool_calls && delta.tool_calls.length > 0) {
        for (const toolCall of delta.tool_calls) {
          const toolCallId = toolCall.id;
          
          // Check if this is a new tool call by ID
          if (toolCallId && toolCallId !== currentToolCallId) {
            // End previous tool call if one was active
            if (isToolUse && currentToolCallId) {
              const contentBlockStop: MessageEvent = {
                type: "content_block_stop",
                index: contentBlockIndex,
              };
              write(
                `event: content_block_stop\ndata: ${JSON.stringify(
                  contentBlockStop
                )}\n\n`
              );
            }

            // Start new tool call block
            isToolUse = true;
            currentToolCallId = toolCallId;
            contentBlockIndex++;
            toolCallJsonMap.set(toolCallId, ""); // Initialize JSON accumulator for this tool call

            const toolBlock: ContentBlock = {
              type: "tool_use",
              id: toolCallId,
              name: toolCall.function?.name,
              input: {},
            };

            const toolBlockStart: MessageEvent = {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: toolBlock,
            };

            currentContentBlocks.push(toolBlock);

            write(
              `event: content_block_start\ndata: ${JSON.stringify(
                toolBlockStart
              )}\n\n`
            );
          }

          // Stream tool call JSON
          if (toolCall.function?.arguments && currentToolCallId) {
            const jsonDelta: MessageEvent = {
              type: "content_block_delta",
              index: contentBlockIndex,
              delta: {
                type: "input_json_delta",
                partial_json: toolCall.function.arguments,
              },
            };

            // Accumulate JSON for this specific tool call
            const currentJson = toolCallJsonMap.get(currentToolCallId) || "";
            toolCallJsonMap.set(currentToolCallId, currentJson + toolCall.function.arguments);
            toolUseJson = toolCallJsonMap.get(currentToolCallId) || "";

            try {
              const parsedJson = JSON.parse(toolUseJson);
              currentContentBlocks[contentBlockIndex].input = parsedJson;
            } catch (e) {
              log("JSON parsing error (continuing to accumulate):", e);
              // JSON not yet complete, continue accumulating
            }

            write(
              `event: content_block_delta\ndata: ${JSON.stringify(jsonDelta)}\n\n`
            );
          }
        }
      } else if (delta.content) {
        // Handle regular text content
        if (isToolUse) {
          log("Tool call ended here:", delta);
          // End previous tool call block
          const contentBlockStop: MessageEvent = {
            type: "content_block_stop",
            index: contentBlockIndex,
          };

          write(
            `event: content_block_stop\ndata: ${JSON.stringify(
              contentBlockStop
            )}\n\n`
          );
          contentBlockIndex++;
          isToolUse = false;
          currentToolCallId = null;
          toolUseJson = ""; // Reset for safety
        }

        if (!delta.content) continue;

        // If text block not yet started, send content_block_start
        if (!hasStartedTextBlock) {
          const textBlock: ContentBlock = {
            type: "text",
            text: "",
          };

          const textBlockStart: MessageEvent = {
            type: "content_block_start",
            index: contentBlockIndex,
            content_block: textBlock,
          };

          currentContentBlocks.push(textBlock);

          write(
            `event: content_block_start\ndata: ${JSON.stringify(
              textBlockStart
            )}\n\n`
          );
          hasStartedTextBlock = true;
        }

        // Send regular text content
        const contentDelta: MessageEvent = {
          type: "content_block_delta",
          index: contentBlockIndex,
          delta: {
            type: "text_delta",
            text: delta.content,
          },
        };

        // Update content block text
        if (currentContentBlocks[contentBlockIndex]) {
          currentContentBlocks[contentBlockIndex].text += delta.content;
        }

        write(
          `event: content_block_delta\ndata: ${JSON.stringify(
            contentDelta
          )}\n\n`
        );
      }
    }
  } catch (e: any) {
    // If text block not yet started, send content_block_start
    if (!hasStartedTextBlock) {
      const textBlock: ContentBlock = {
        type: "text",
        text: "",
      };

      const textBlockStart: MessageEvent = {
        type: "content_block_start",
        index: contentBlockIndex,
        content_block: textBlock,
      };

      currentContentBlocks.push(textBlock);

      write(
        `event: content_block_start\ndata: ${JSON.stringify(
          textBlockStart
        )}\n\n`
      );
      hasStartedTextBlock = true;
    }

    // Send regular text content
    const contentDelta: MessageEvent = {
      type: "content_block_delta",
      index: contentBlockIndex,
      delta: {
        type: "text_delta",
        text: JSON.stringify(e),
      },
    };

    // Update content block text
    if (currentContentBlocks[contentBlockIndex]) {
      currentContentBlocks[contentBlockIndex].text += JSON.stringify(e);
    }

    write(
      `event: content_block_delta\ndata: ${JSON.stringify(contentDelta)}\n\n`
    );
  }

  // Close last content block if any is open
  if (isToolUse || hasStartedTextBlock) {
    const contentBlockStop: MessageEvent = {
      type: "content_block_stop",
      index: contentBlockIndex,
    };

    write(
      `event: content_block_stop\ndata: ${JSON.stringify(contentBlockStop)}\n\n`
    );
  }

  // Send message_delta event with appropriate stop_reason
  const messageDelta: MessageEvent = {
    type: "message_delta",
    delta: {
      stop_reason: isToolUse ? "tool_use" : "end_turn",
      stop_sequence: null,
      content: currentContentBlocks,
    },
    usage: { input_tokens: 100, output_tokens: 150 },
  };
  if (!isToolUse) {
    log("body: ", body, "messageDelta: ", messageDelta);
  }

  write(`event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`);

  // Send message_stop event
  const messageStop: MessageEvent = {
    type: "message_stop",
  };

  write(`event: message_stop\ndata: ${JSON.stringify(messageStop)}\n\n`);
  res.end();
}
