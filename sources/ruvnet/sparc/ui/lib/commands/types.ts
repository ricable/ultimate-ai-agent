export type SubmitParams = {
  messages: Array<{
    role: 'user' | 'assistant'
    content: Array<{
      type: string
      text: string
      icon?: string
    }>
    loading?: boolean
    streaming?: boolean
  }>
  userID: string
  template: any
  model: any
  config: any
  clearInput?: boolean
  updateLast?: boolean
}

export type SubmitFunction = (params: SubmitParams) => void

export type CommandContext = {
  userID: string
  template: any
  model: any
  config: any
  messages: Array<{
    role: 'user' | 'assistant'
    content: Array<{
      type: string
      text: string
      icon?: string
    }>
    loading?: boolean
    streaming?: boolean
  }>
  defaultHandler: CommandHandler
}

export type CommandHandler = (args: string, submit: SubmitFunction, context: CommandContext) => Promise<boolean>

export interface Command {
  name: string
  description: string
  handler: CommandHandler
  preview?: (fragment: any, result: any) => {
    title: string
    description: string
    files: Array<{
      name: string
      content: string
    }>
  }
}

export type CommandRegistry = Record<string, Command>
