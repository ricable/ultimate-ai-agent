import { CommandHandler } from './types'

export const fragment: CommandHandler = async (args: string, submit, context) => {
  if (!args) return false

  // Handle regular input without command prefix
  const content = [{ type: 'text', text: args }]

  // Add user message to chat
  submit({
    messages: [{
      role: 'user',
      content,
    }],
    userID: context.userID,
    model: context.model,
    template: context.template,
    config: context.config,
    clearInput: true
  })

  // Submit for AI processing
  submit({
    userID: context.userID,
    messages: [{
      role: 'user',
      content: [{ type: 'text', text: args }]
    }],
    template: context.template,
    model: context.model,
    config: context.config
  })

  return true
}
