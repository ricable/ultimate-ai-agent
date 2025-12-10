#!/usr/bin/env python3
"""
MLX Deep Council Worker

Distributed inference worker for the MLX Deep Council system.
Runs on each Mac node and provides local LLM inference capabilities.

This script is designed to be launched via `mlx.launch` for distributed
coordination across multiple Mac machines.

Features:
- MLX-native inference with Apple Silicon optimization
- Distributed communication using MLX ring topology
- HTTP API compatible with OpenAI format
- Streaming response support
- Health monitoring and metrics

Usage:
    mlx.launch --hostfile hosts.json mlx_council_worker.py --model mlx-community/Llama-3.2-3B-Instruct-4bit

Requirements:
    - mlx >= 0.20.0
    - mlx-lm >= 0.18.0
    - fastapi
    - uvicorn
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Rank %(rank)s] %(levelname)s - %(message)s',
)


# ============================================================================
# DISTRIBUTED CONTEXT
# ============================================================================

@dataclass
class DistributedContext:
    """Manages MLX distributed group context."""
    world: Any = None
    rank: int = 0
    size: int = 1
    is_chairman: bool = False

    def __post_init__(self):
        # Initialize distributed group
        try:
            self.world = mx.distributed.init()
            self.rank = self.world.rank()
            self.size = self.world.size()
            self.is_chairman = self.rank == 0

            # Update logging format with rank
            for handler in logging.root.handlers:
                handler.setFormatter(
                    logging.Formatter(
                        f'%(asctime)s - [Rank {self.rank}] %(levelname)s - %(message)s'
                    )
                )
        except Exception as e:
            logging.warning(f"Distributed init failed, running standalone: {e}")

    def barrier(self):
        """Synchronization barrier across all ranks."""
        if self.size > 1:
            # Use all_sum as a barrier
            mx.distributed.all_sum(mx.ones(1))

    def all_gather_strings(self, data: str) -> List[str]:
        """Gather strings from all ranks."""
        if self.size == 1:
            return [data]

        # Encode string to bytes, pad to max length
        encoded = data.encode('utf-8')
        max_len = 65536  # Max message size

        if len(encoded) > max_len:
            encoded = encoded[:max_len]

        # Pad to fixed length
        padded = encoded + b'\x00' * (max_len - len(encoded))

        # Convert to MLX array
        arr = mx.array(list(padded), dtype=mx.uint8)

        # Gather from all ranks
        gathered = mx.distributed.all_gather(arr)

        # Decode results
        results = []
        for i in range(self.size):
            chunk = gathered[i * max_len:(i + 1) * max_len]
            bytes_data = bytes(chunk.tolist())
            # Find null terminator
            null_idx = bytes_data.find(b'\x00')
            if null_idx > 0:
                bytes_data = bytes_data[:null_idx]
            results.append(bytes_data.decode('utf-8'))

        return results

    def broadcast_from_rank(self, data: str, source_rank: int = 0) -> str:
        """Broadcast string from source rank to all others."""
        if self.size == 1:
            return data

        if self.rank == source_rank:
            # Sender: gather but only use own data
            return self.all_gather_strings(data)[source_rank]
        else:
            # Receivers: gather and take from source
            return self.all_gather_strings("")[source_rank]


# ============================================================================
# MODEL MANAGER
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a loaded model."""
    name: str
    path: str
    context_length: int = 4096
    quantization: str = "4bit"
    loaded: bool = False
    memory_used: int = 0


class ModelManager:
    """Manages MLX model loading and inference."""

    def __init__(self, context: DistributedContext):
        self.context = context
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.current_model: Optional[str] = None

    def load_model(self, model_path: str, name: Optional[str] = None) -> str:
        """Load a model from HuggingFace or local path."""
        try:
            from mlx_lm import load

            model_name = name or Path(model_path).name
            logging.info(f"Loading model: {model_path}")

            start_time = time.time()
            model, tokenizer = load(model_path)
            load_time = time.time() - start_time

            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.configs[model_name] = ModelConfig(
                name=model_name,
                path=model_path,
                loaded=True,
            )
            self.current_model = model_name

            logging.info(f"Model loaded in {load_time:.2f}s")
            return model_name

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using the specified model."""
        from mlx_lm import generate

        model_name = model_name or self.current_model
        if not model_name or model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )

        return response

    def generate_stream(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation using the specified model."""
        from mlx_lm import stream_generate

        model_name = model_name or self.current_model
        if not model_name or model_name not in self.models:
            raise ValueError(f"Model not loaded: {model_name}")

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        ):
            yield chunk


# ============================================================================
# COUNCIL WORKER
# ============================================================================

@dataclass
class WorkerMetrics:
    """Metrics for the council worker."""
    requests_processed: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def average_latency(self) -> float:
        if self.requests_processed == 0:
            return 0
        return self.total_latency_ms / self.requests_processed

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()


class CouncilWorker:
    """
    Council worker that handles distributed inference.

    Each worker runs on a Mac in the council and processes queries
    as part of the three-stage council process.
    """

    def __init__(
        self,
        context: DistributedContext,
        model_manager: ModelManager,
        member_id: Optional[str] = None,
    ):
        self.context = context
        self.model_manager = model_manager
        self.member_id = member_id or f"member_{context.rank}"
        self.metrics = WorkerMetrics()
        self.anonymous_id = f"Model {chr(65 + context.rank)}"  # A, B, C, etc.

    async def process_query(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Stage 1: Process a query and generate individual response.
        """
        start_time = time.time()

        try:
            # Build prompt
            if system_prompt:
                full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"
            else:
                full_prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"

            # Generate response
            response = self.model_manager.generate(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            latency = (time.time() - start_time) * 1000
            self.metrics.requests_processed += 1
            self.metrics.total_latency_ms += latency

            # Extract confidence (simple heuristic)
            confidence = 0.75
            if "confident" in response.lower():
                confidence = 0.85
            elif "uncertain" in response.lower():
                confidence = 0.5

            return {
                "member_id": self.member_id,
                "anonymous_id": self.anonymous_id,
                "content": response,
                "confidence": confidence,
                "latency_ms": latency,
                "token_count": len(response.split()),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.metrics.errors += 1
            logging.error(f"Query processing failed: {e}")
            raise

    async def conduct_peer_review(
        self,
        original_query: str,
        responses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Stage 2: Review and rank other members' responses.
        """
        # Format responses for review (excluding own response)
        review_text = "ORIGINAL QUERY:\n" + original_query + "\n\n"
        review_text += "RESPONSES TO EVALUATE:\n\n"

        for resp in responses:
            if resp["member_id"] != self.member_id:
                review_text += f"=== {resp['anonymous_id']} ===\n"
                review_text += f"{resp['content']}\n\n"

        review_prompt = f"""You are a peer reviewer. Evaluate the following responses to a query.

{review_text}

For each response, provide:
1. Score (1-10)
2. Key strengths
3. Key weaknesses
4. Your ranking

Format:
EVALUATION FOR [Model ID]:
Score: [1-10]
Strengths: [list]
Weaknesses: [list]

OVERALL RANKING: [best to worst]
PREFER OTHER: [Yes/No, and why if yes]"""

        response = self.model_manager.generate(
            review_prompt,
            temperature=0.3,
            max_tokens=2048,
        )

        # Parse review (simplified)
        reviews = []
        for resp in responses:
            if resp["member_id"] != self.member_id:
                # Extract score for this response
                score = 7  # Default
                import re
                score_match = re.search(
                    rf"{resp['anonymous_id']}.*?Score:\s*(\d+)",
                    response,
                    re.IGNORECASE | re.DOTALL
                )
                if score_match:
                    score = min(10, max(1, int(score_match.group(1))))

                reviews.append({
                    "anonymous_id": resp["anonymous_id"],
                    "score": score,
                    "reasoning": "",
                })

        # Check for self-deferral
        prefer_other = "prefer other: yes" in response.lower()

        return {
            "reviewer_id": self.member_id,
            "reviews": reviews,
            "self_evaluation": {
                "prefer_other": prefer_other,
            },
            "raw_review": response,
        }

    async def synthesize_as_chairman(
        self,
        query: str,
        responses: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        aggregated_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Stage 3: Synthesize final response as chairman.
        """
        if not self.context.is_chairman:
            raise ValueError("Only the chairman can synthesize")

        # Sort responses by score
        sorted_responses = sorted(
            responses,
            key=lambda r: aggregated_scores.get(r["anonymous_id"], 0),
            reverse=True
        )

        synthesis_prompt = f"""You are the Chairman of an AI council. Synthesize the best response from council members.

ORIGINAL QUERY:
{query}

COUNCIL RESPONSES (ordered by peer review score):

"""
        for resp in sorted_responses:
            score = aggregated_scores.get(resp["anonymous_id"], 0)
            synthesis_prompt += f"=== {resp['anonymous_id']} (Score: {score:.1f}/10) ===\n"
            synthesis_prompt += f"{resp['content']}\n\n"

        synthesis_prompt += """
Create a comprehensive final response that:
1. Uses the best insights from top-rated responses
2. Resolves any conflicts between responses
3. Provides an authoritative, well-reasoned answer

Format your response as:
FINAL RESPONSE:
[Your synthesized response]

REASONING:
[How you combined the responses]

CONFIDENCE: [0-100%]"""

        response = self.model_manager.generate(
            synthesis_prompt,
            temperature=0.5,
            max_tokens=4096,
        )

        # Parse synthesis
        import re

        # Extract final response
        final_match = re.search(
            r"FINAL RESPONSE:\s*(.*?)(?=REASONING:|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        final_response = final_match.group(1).strip() if final_match else response

        # Extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.*?)(?=CONFIDENCE:|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract confidence
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", response, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) / 100 if confidence_match else 0.8

        return {
            "final_response": final_response,
            "reasoning": reasoning,
            "confidence_score": confidence,
            "sources_used": [r["anonymous_id"] for r in sorted_responses[:3]],
            "conflicts_resolved": [],
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics."""
        return {
            "member_id": self.member_id,
            "rank": self.context.rank,
            "is_chairman": self.context.is_chairman,
            "requests_processed": self.metrics.requests_processed,
            "average_latency_ms": self.metrics.average_latency,
            "errors": self.metrics.errors,
            "uptime_seconds": self.metrics.uptime_seconds,
        }


# ============================================================================
# HTTP API SERVER
# ============================================================================

def create_api_app(worker: CouncilWorker):
    """Create FastAPI application for the council worker."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(
        title="MLX Council Worker",
        description="Distributed LLM inference worker for MLX Deep Council",
        version="1.0.0",
    )

    class Message(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "default"
        messages: List[Message]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 2048
        stream: Optional[bool] = False

    class CouncilQueryRequest(BaseModel):
        query: str
        system_prompt: Optional[str] = None
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 2048

    class PeerReviewRequest(BaseModel):
        original_query: str
        responses: List[Dict[str, Any]]

    class SynthesisRequest(BaseModel):
        query: str
        responses: List[Dict[str, Any]]
        reviews: List[Dict[str, Any]]
        aggregated_scores: Dict[str, float]

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "rank": worker.context.rank,
            "is_chairman": worker.context.is_chairman,
        }

    @app.get("/v1/models")
    async def list_models():
        models = list(worker.model_manager.configs.keys())
        return {
            "data": [
                {"id": m, "object": "model", "owned_by": "local"}
                for m in models
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""
        # Build prompt from messages
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}<|end|>\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}<|end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}<|end|>\n"
        prompt += "<|assistant|>\n"

        if request.stream:
            async def generate():
                for chunk in worker.model_manager.generate_stream(
                    prompt,
                    max_tokens=request.max_tokens or 2048,
                    temperature=request.temperature or 0.7,
                ):
                    data = {
                        "choices": [{
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )

        response = worker.model_manager.generate(
            prompt,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 0.7,
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            }
        }

    @app.post("/council/query")
    async def council_query(request: CouncilQueryRequest):
        """Stage 1: Process individual query."""
        result = await worker.process_query(
            request.query,
            system_prompt=request.system_prompt,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens or 2048,
        )
        return result

    @app.post("/council/review")
    async def council_review(request: PeerReviewRequest):
        """Stage 2: Conduct peer review."""
        result = await worker.conduct_peer_review(
            request.original_query,
            request.responses,
        )
        return result

    @app.post("/council/synthesize")
    async def council_synthesize(request: SynthesisRequest):
        """Stage 3: Chairman synthesis."""
        if not worker.context.is_chairman:
            raise HTTPException(
                status_code=400,
                detail="Only the chairman can synthesize"
            )

        result = await worker.synthesize_as_chairman(
            request.query,
            request.responses,
            request.reviews,
            request.aggregated_scores,
        )
        return result

    @app.get("/metrics")
    async def get_metrics():
        return worker.get_metrics()

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLX Deep Council Worker"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Model to load (HuggingFace path or local path)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the API server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the API server to",
    )
    parser.add_argument(
        "--member-id",
        type=str,
        default=None,
        help="Custom member ID for this worker",
    )

    args = parser.parse_args()

    # Initialize distributed context
    context = DistributedContext()
    logging.info(f"Initialized as rank {context.rank} of {context.size}")

    if context.is_chairman:
        logging.info("This node is the CHAIRMAN")

    # Initialize model manager
    model_manager = ModelManager(context)

    # Load model
    logging.info(f"Loading model: {args.model}")
    model_manager.load_model(args.model)

    # Create worker
    worker = CouncilWorker(
        context=context,
        model_manager=model_manager,
        member_id=args.member_id,
    )

    # Adjust port based on rank (to avoid conflicts on same machine)
    port = args.port + context.rank

    # Create and run API
    app = create_api_app(worker)

    logging.info(f"Starting API server on {args.host}:{port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
