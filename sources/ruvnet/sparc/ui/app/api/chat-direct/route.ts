import { StreamingTextResponse, Message } from 'ai'
import { NextResponse } from 'next/server'
import { ChatAnthropic } from '@langchain/anthropic'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { AIMessage } from '@langchain/core/messages'
import { MessageContentText } from '@langchain/core/messages'

export async function POST(req: Request) {
  try {
    const { prompt, messages: previousMessages, modelName } = await req.json()
    
    const model = new ChatAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      modelName: modelName || 'claude-3-sonnet-20240229',
      streaming: true
    })

    // Convert previous messages to the format expected by the model
    const messageHistory = previousMessages?.map((msg: any) => {
      // Handle nested content structure
      let messageText = ''
      if (Array.isArray(msg.content)) {
        const textContent = msg.content.find((c: any) => c.type === 'text')
        messageText = textContent?.text || ''
      } else {
        messageText = msg.content || ''
      }
      
      return msg.role === 'system'
        ? new SystemMessage(messageText)
        : new HumanMessage(messageText)
    }) || []

    // Use the full message history
    const messages = messageHistory

    const stream = await model.stream(messages);
    
    // Transform the stream to emit text chunks
    const textEncoder = new TextEncoder()
    const textStream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            // Handle both string and complex message content
            let content = ''
            
            if (chunk instanceof AIMessage) {
              if (typeof chunk.content === 'string') {
                content = chunk.content
              } else if (Array.isArray(chunk.content)) {
                content = chunk.content
                  .filter((c): c is MessageContentText => c.type === 'text')
                  .map(c => c.text)
                  .join('')
              }
            } else if (typeof chunk.content === 'string') {
              content = chunk.content
            }

            if (content) {
              controller.enqueue(textEncoder.encode(content))
            }
          }
          controller.close()
        } catch (error) {
          controller.error(error)
        }
      }
    })

    return new StreamingTextResponse(textStream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8'
      }
    })
    
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    )
  }
}
