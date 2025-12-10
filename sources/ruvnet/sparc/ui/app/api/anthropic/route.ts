import { NextResponse } from 'next/server'
import { ChatAnthropic } from '@langchain/anthropic'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'

interface AnthropicMessage {
  role: 'system' | 'user'
  content: string
}

export async function POST(req: Request) {
  try {
    const { prompt, messages: inputMessages, modelName } = await req.json()
    
    const model = new ChatAnthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
      modelName: modelName || 'claude-3-sonnet-20240229'
    })

    let messages = []
    
    if (inputMessages) {
      // Handle complex message structure
      messages = inputMessages.map((msg: AnthropicMessage) => {
        let content = '';
        
        // Handle deeply nested text structure
        if (Array.isArray(msg.content)) {
          const textContent = msg.content.find(c => c.type === 'text');
          if (textContent && textContent.text) {
            content = textContent.text.text || '';
          }
        } else if (typeof msg.content === 'string') {
          content = msg.content;
        }
        
        if (msg.role === 'system') {
          return new SystemMessage(content)
        }
        return new HumanMessage(content)
      })
    } else {
      // Handle simple prompt format
      messages = [
        new SystemMessage(`You are a research assistant. Analyze the provided topic and generate a comprehensive research report.`),
        new HumanMessage(prompt)
      ]
    }

    const response = await model.invoke(messages)
    
    return NextResponse.json({ 
      content: response.content.toString() 
    })
    
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    )
  }
}
