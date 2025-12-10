export async function POST(req: Request) {
  // Simply return success - actual response is handled by the research command
  return new Response(JSON.stringify({ success: true }), {
    headers: {
      'Content-Type': 'application/json'
    }
  })
}
