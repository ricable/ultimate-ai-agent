import { CoreMessage } from 'ai'

export async function POST(req: Request) {
  // We're not using these but keep the type checking
  const {
    messages,
    userID,
    model,
    config,
  }: {
    messages: CoreMessage[]
    userID: string
    model: any
    config: any
  } = await req.json()

  // Simply return success - the actual response is handled by the research command
  return new Response(JSON.stringify({ success: true }), {
    headers: {
      'Content-Type': 'application/json'
    }
  })
}
