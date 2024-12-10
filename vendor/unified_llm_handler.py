from typing import (
    Dict,
    List,
    Union,
    Type,
    TypeVar,
    Optional,
    Sequence,
    Literal,
    Generic,
)
from datetime import datetime
import os
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
import logfire
from aiolimiter import AsyncLimiter
import traceback

from pydantic_ai import Agent
from pydantic_ai.models import Model, KnownModelName, infer_model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.exceptions import UserError

logfire.configure(send_to_logfire="if-token-present")

T = TypeVar("T", bound=BaseModel)
ModelProvider = Literal["openai", "groq", "vertexai"]


class BatchMetadata(BaseModel):
    """Metadata for batch processing jobs."""

    batch_id: str
    input_file_id: str
    status: str
    created_at: datetime
    last_updated: datetime
    error: Optional[str] = None
    output_file_path: Optional[str] = None
    num_requests: int


class BatchResult(BaseModel):
    """Results from batch processing."""

    metadata: BatchMetadata
    results: List[Dict[str, Union[str, BaseModel]]]


class SimpleResponse(BaseModel):
    """Simple response model for testing."""

    content: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class MathResponse(BaseModel):
    """Response model for math problems."""

    answer: Optional[float] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)


class PersonResponse(BaseModel):
    """Response model for person descriptions."""

    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    occupation: Optional[str] = None
    skills: Optional[List[str]] = None


class UnifiedResponse(BaseModel, Generic[T]):
    """A unified response envelope."""

    success: bool
    data: Optional[Union[T, List[T], BatchResult]] = None
    error: Optional[str] = None
    original_prompt: Optional[str] = None


class CityLocation(BaseModel):
    city: str
    country: str


class UnifiedLLMHandler:
    MODEL_PREFIXES = {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4-turbo": "openai",
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "o1-preview": "openai",
        "o1-mini": "openai",
        "llama-3.1-70b-versatile": "groq",
        "llama3-groq-70b-8192-tool-use-preview": "groq",
        "llama-3.1-70b-specdec": "groq",
        "llama-3.1-8b-instant": "groq",
        "llama-3.2-1b-preview": "groq",
        "llama-3.2-3b-preview": "groq",
        "mixtral-8x7b-32768": "groq",
        "gemma2-9b-it": "groq",
        "gemma-7b-it": "groq",
        "gemini-1.5-flash": None,
        "gemini-1.5-pro": None,
        "gemini-1.0-pro": None,
        "gemini-1.5-flash-8b": None,
        "vertex-gemini-1.5-flash": "vertexai",
        "vertex-gemini-1.5-pro": "vertexai",
        "vertex-gemini-1.5-flash-8b": "vertexai",
    }

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        batch_output_dir: str = "batch_output",
    ):
        self.rate_limiter = (
            AsyncLimiter(requests_per_minute, 60) if requests_per_minute else None
        )
        self.batch_output_dir = Path(batch_output_dir)
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_prefixed_model_name(self, model_name: str) -> str:
        if isinstance(model_name, Model):
            return model_name
        if ":" in model_name:
            return model_name
        prefix = self.MODEL_PREFIXES.get(model_name)
        return model_name if prefix is None else f"{prefix}:{model_name}"

    async def process(
        self,
        prompts: Union[str, List[str]],
        model: Union[str, KnownModelName, Model],
        response_type: Type[T],
        *,
        system_message: Union[str, Sequence[str]] = (),
        batch_size: int = 1000,
        batch_mode: bool = False,
        retries: int = 1,
    ) -> UnifiedResponse[Union[T, List[T], BatchResult]]:
        with logfire.span("llm_processing"):
            original_prompt_for_error = None
            if isinstance(prompts, str):
                original_prompt_for_error = prompts
            elif isinstance(prompts, list) and prompts:
                original_prompt_for_error = prompts[0]

            try:
                if prompts is None:
                    raise UserError("Prompts cannot be None.")
                if isinstance(prompts, list) and len(prompts) == 0:
                    raise UserError("Prompts list cannot be empty.")

                if isinstance(model, str):
                    model = self._get_prefixed_model_name(model)

                try:
                    model_instance = (
                        infer_model(model)
                        if isinstance(model, (str, KnownModelName))
                        else model
                    )
                except Exception as e:
                    raise UserError(f"Invalid model name: {model}. Error: {e}")

                agent = Agent(
                    model_instance,
                    result_type=response_type,
                    system_prompt=system_message,
                    retries=retries,
                )

                if batch_mode:
                    if not isinstance(model_instance, OpenAIModel):
                        raise UserError(
                            "Batch API mode is only supported for OpenAI models."
                        )
                    batch_result = await self._process_batch(
                        agent, prompts, response_type
                    )
                    return UnifiedResponse(success=True, data=batch_result)

                if isinstance(prompts, str):
                    data = await self._process_single(agent, prompts)
                    return UnifiedResponse(success=True, data=data)
                else:
                    data = await self._process_multiple(agent, prompts, batch_size)
                    return UnifiedResponse(success=True, data=data)

            except UserError as e:
                full_trace = traceback.format_exc()
                error_msg = f"UserError: {e}\nFull Traceback:\n{full_trace}"
                with logfire.span(
                    "error_handling", error=str(e), error_type="user_error"
                ):
                    return UnifiedResponse(
                        success=False,
                        error=error_msg,
                        original_prompt=original_prompt_for_error,
                    )
            except Exception as e:
                full_trace = traceback.format_exc()
                error_msg = f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                with logfire.span(
                    "error_handling", error=str(e), error_type="unexpected_error"
                ):
                    return UnifiedResponse(
                        success=False,
                        error=error_msg,
                        original_prompt=original_prompt_for_error,
                    )

    async def _process_single(self, agent: Agent, prompt: str) -> T:
        with logfire.span("process_single"):
            if self.rate_limiter:
                async with self.rate_limiter:
                    result = await agent.run(prompt)
            else:
                result = await agent.run(prompt)
            return result.data

    async def _process_multiple(
        self, agent: Agent, prompts: List[str], batch_size: int
    ) -> List[T]:
        results = []
        with logfire.span("process_multiple"):
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]

                async def process_prompt(p: str) -> T:
                    if self.rate_limiter:
                        async with self.rate_limiter:
                            res = await agent.run(p)
                    else:
                        res = await agent.run(p)
                    return res.data

                batch_results = await asyncio.gather(
                    *(process_prompt(p) for p in batch)
                )
                results.extend(batch_results)
        return results

    async def _process_batch(
        self, agent: Agent, prompts: List[str], response_type: Type[T]
    ) -> BatchResult:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = self.batch_output_dir / f"batch_{timestamp}.jsonl"

        with logfire.span("process_batch"):
            with batch_file.open("w") as f:
                for i, prompt in enumerate(prompts):
                    request = {
                        "custom_id": f"req_{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": agent.model.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    }
                    f.write(json.dumps(request) + "\n")

            batch_upload = await agent.model.client.files.create(
                file=batch_file.open("rb"), purpose="batch"
            )

            batch = await agent.model.client.batches.create(
                input_file_id=batch_upload.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            metadata = BatchMetadata(
                batch_id=batch.id,
                input_file_id=batch_upload.id,
                status="in_progress",
                created_at=datetime.now(),
                last_updated=datetime.now(),
                num_requests=len(prompts),
            )

            while True:
                status = await agent.model.client.batches.retrieve(batch.id)
                metadata.status = status.status
                metadata.last_updated = datetime.now()

                if status.status == "completed":
                    break
                elif status.status in ["failed", "canceled"]:
                    metadata.error = f"Batch failed with status: {status.status}"
                    return BatchResult(metadata=metadata, results=[])

                await asyncio.sleep(10)

            output_file = self.batch_output_dir / f"batch_{batch.id}_results.jsonl"
            result_content = await agent.model.client.files.content(
                status.output_file_id
            )

            with output_file.open("wb") as f:
                f.write(result_content.content)

            metadata.output_file_path = str(output_file)

            results = []
            with output_file.open() as f:
                for line, prompt in zip(f, prompts):
                    data = json.loads(line)
                    try:
                        content = data["response"]["body"]["choices"][0]["message"][
                            "content"
                        ]
                        r = response_type.construct()
                        if "content" in response_type.model_fields:
                            setattr(r, "content", content)
                        if "confidence" in response_type.model_fields:
                            setattr(r, "confidence", 0.95)
                        results.append({"prompt": prompt, "response": r})
                    except Exception as e:
                        full_trace = traceback.format_exc()
                        error_msg = (
                            f"Unexpected error: {e}\nFull Traceback:\n{full_trace}"
                        )
                        results.append({"prompt": prompt, "error": error_msg})

            return BatchResult(metadata=metadata, results=results)


async def run_tests():
    """Run test scenarios to demonstrate usage of UnifiedLLMHandler."""
    handler = UnifiedLLMHandler(requests_per_minute=2000)

    # Test 1: Single prompt
    single_result = await handler.process(
        "Explain quantum computing in simple terms", "gpt-4o-mini", SimpleResponse
    )
    print("\nTest 1 (Single Prompt):")
    print("Success?", single_result.success)
    print(
        "Data:",
        (
            single_result.data.model_dump(mode="python")
            if single_result.success and single_result.data
            else single_result.error
        ),
    )

    # Test 2: Multiple prompts
    math_questions = ["What is 127+345?", "Calculate 15% of 2500", "What is sqrt(169)?"]
    multi_result = await handler.process(
        math_questions,
        "gemini-1.5-flash-8b",
        MathResponse,
        system_message="You are a precise mathematical assistant. Always show your reasoning.",
    )
    print("\nTest 2 (Multiple Prompts):")
    print("Success?", multi_result.success)
    if multi_result.success and multi_result.data:
        if isinstance(multi_result.data, list):
            for d in multi_result.data:
                print(d.model_dump(mode="python"))
        else:
            print(multi_result.data.model_dump(mode="python"))
    else:
        print(multi_result.error)

    # Test 3: Parallel processing
    person_prompts = [
        "Describe a software engineer in Silicon Valley",
        "Describe a chef in a Michelin star restaurant",
    ]
    person_result = await handler.process(
        person_prompts, "gpt-4o-mini", PersonResponse, batch_size=2
    )
    print("\nTest 3 (Parallel Processing):")
    print("Success?", person_result.success)
    if person_result.success and isinstance(person_result.data, list):
        for d in person_result.data:
            print(d.model_dump(mode="python"))
    else:
        print(person_result.error)

    # Test 4: Batch API processing with invalid model
    simple_prompts = [f"Write a one-sentence story about number {i}" for i in range(3)]
    batch_result = await handler.process(
        simple_prompts,
        "gemini-1.5-flash-8b",  # Invalid for batch API
        SimpleResponse,
        batch_mode=True,
    )
    print("\nTest 4 (Batch API with invalid model):")
    print("Success?", batch_result.success)
    print(
        "Output:",
        (
            batch_result.error
            if not batch_result.success
            else batch_result.data.model_dump(mode="python")
        ),
    )

    # Test 5: Invalid model
    invalid_model_result = await handler.process(
        "Test prompt", "invalid-model", SimpleResponse
    )
    print("\nTest 5 (Invalid Model):")
    print("Success?", invalid_model_result.success)
    print(
        "Output:",
        (
            invalid_model_result.error
            if not invalid_model_result.success
            else invalid_model_result.data.model_dump(mode="python")
        ),
    )

    # Test 6: Invalid prompt (None)
    invalid_prompt_result = await handler.process(None, "gpt-4o-mini", SimpleResponse)
    print("\nTest 6 (Invalid Prompt):")
    print("Success?", invalid_prompt_result.success)
    print(
        "Output:",
        (
            invalid_prompt_result.error
            if not invalid_prompt_result.success
            else invalid_prompt_result.data.model_dump(mode="python")
        ),
    )

    # Test 7: Ollama model integration
    # This test verifies the ollama:llama3.2 model works as expected.
    ollama_result = await handler.process(
        "Where the Olympics held in 2012?", "ollama:llama3.2", CityLocation
    )
    print("\nTest 7 (Ollama Model):")
    print("Success?", ollama_result.success)
    if ollama_result.success and ollama_result.data:
        print("Data:", ollama_result.data.model_dump(mode="python"))
    else:
        print("Error:", ollama_result.error)


if __name__ == "__main__":
    asyncio.run(run_tests())
