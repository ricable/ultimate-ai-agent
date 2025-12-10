context7project


Tools (2)
Resources (0)
Errors (1)
resolve-library-id
Resolves a package/product name to a Context7-compatible library ID and returns a list of matching libraries. You MUST call this function before 'get-library-docs' to obtain a valid Context7-compatible library ID UNLESS the user explicitly provides a library ID in the format '/org/project' or '/org/project/version' in their query. Selection Process: 1. Analyze the query to understand what library/package the user is looking for 2. Return the most relevant match based on: - Name similarity to the query (exact matches prioritized) - Description relevance to the query's intent - Documentation coverage (prioritize libraries with higher Code Snippet counts) - Trust score (consider libraries with scores of 7-10 more authoritative) Response Format: - Return the selected library ID in a clearly marked section - Provide a brief explanation for why this library was chosen - If multiple good matches exist, acknowledge this but proceed with the most relevant one - If no good matches exist, clearly state this and suggest query refinements For ambiguous queries, request clarification before proceeding with a best-guess match.
Parameters
libraryName*
Library name to search for and retrieve a Context7-compatible library ID.
get-library-docs
Fetches up-to-date documentation for a library. You must call 'resolve-library-id' first to obtain the exact Context7-compatible library ID required to use this tool, UNLESS the user explicitly provides a library ID in the format '/org/project' or '/org/project/version' in their query.
Parameters
context7CompatibleLibraryID*
Exact Context7-compatible library ID (e.g., '/mongodb/docs', '/vercel/next.js', '/supabase/supabase', '/vercel/next.js/v14.3.0-canary.87') retrieved from 'resolve-library-id' or directly from user query in the format '/org/project' or '/org/project/version'.
topic
Topic to focus documentation on (e.g., 'hooks', 'routing').
tokens
Maximum number of tokens of documentation to retrieve (default: 10000). Higher values provide more context but consume more tokens.
Network Timeout

1 minute
Maximum time to wait for server responses
preplexityproject


Tools (1)
Resources (0)
Errors (0)
PERPLEXITYAI_PERPLEXITY_AI_SEARCH
Perplexity ai search interfaces with perplexity ai to perform search queries and return responses from a range of models. this action manages requests to perplexity ai and processes the resulting completions, which may include text, citations, and images based on selected models and settings. key features include: autoprompting to enhance and refine queries, choice of ai models for various content and performance requirements, temperature settings to manage response randomness, top k and top p filters to fine-tune response generation. beta features include citations and images in results, and response streaming for dynamic interaction. note: the parameters 'presence penalty' and 'frequency penalty' are mutually exclusive and cannot be used simultaneously.
Parameters
frequency_penalty
Multiplicative penalty for new tokens based on their frequency in the text to avoid repetition. Mutually exclusive with the 'presence_penalty' parameter. For example: 0.5, 1.0, 1.5.
max_tokens
The maximum number of tokens to generate. Sum of max_tokens and prompt tokens should not exceed the model's context window limit. Unspecified leads to generation until stop token or context window end. For e.g 100, 150, 200
model
The name of the model to use for generating completions. Choose a model based on the desired balance between performance and resource usage. For more information check https://docs.perplexity.ai/guides/model-cards
presence_penalty
Penalty for new tokens based on their current presence in the text, encouraging topic variety. Mutually exclusive with the 'frequency_penalty' parameter. For example: -2.0, 0.0, 2.0.
return_citations
Whether to include citations in the model's response. Citations feature is in closed beta. For e.g True, False
return_images
Whether to include images in the model's response. Image generation feature is in closed beta. For e.g True, False
stream
Whether to stream the response incrementally using server-sent events. For e.g True, False
systemContent*
The system's Content for specifying instructions. For e.g Be precise and concise., Be elaborate and descriptive
temperature
Controls generation randomness, with 0 being deterministic and values approaching 2 being more random. For e.g 0.0, 0.7, 1.5
top_k
Limits the number of high-probability tokens to consider for generation. Set to 0 to disable. For e.g 0, 40, 80
top_p
Nucleus sampling threshold, controlling the token selection pool based on cumulative probability. For e.g 0.1, 0.9, 1.0
userContent*
The user's Content for asking questions or providing input. For e.g How many stars are there in our galaxy?