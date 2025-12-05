"""
LLM service abstraction layer.
Supports multiple LLM providers: Ollama, OpenAI, Anthropic, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator
import httpx
import json
from app.config import get_settings

settings = get_settings()


class LLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
        
        Returns:
            Generated text (or iterator if streaming)
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass


class OllamaLLM(LLMService):
    """
    Ollama LLM service for local models.
    Supports phi3.5, llama3, mistral, etc.
    """
    
    def __init__(
        self, 
        model_name: str = None, 
        base_url: str = None,
        timeout: int = 120
    ):
        self._model_name = model_name or settings.LLM_MODEL
        self._base_url = base_url or settings.OLLAMA_URL
        self._timeout = timeout
        
        print(f"✓ Configured Ollama LLM: {self._model_name}")
        print(f"  Endpoint: {self._base_url}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        top_p: float = 0.9
    ) -> str | Iterator[str]:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt
            max_tokens: Max tokens
            temperature: Sampling temperature
            stream: Whether to stream
            top_p: Nucleus sampling
        
        Returns:
            Generated text or iterator
        """
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        
        if stream:
            return self._generate_stream(payload)
        else:
            return self._generate_blocking(payload)
    
    def _generate_blocking(self, payload: Dict[str, Any]) -> str:
        """Generate text without streaming."""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(self._base_url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
        
        except httpx.TimeoutException:
            return "Error: LLM request timed out. Please try again."
        except httpx.HTTPError as e:
            return f"Error: LLM request failed - {str(e)}"
        except Exception as e:
            return f"Error: Unexpected error - {str(e)}"
    
    def _generate_stream(self, payload: Dict[str, Any]) -> Iterator[str]:
        """Generate text with streaming."""
        try:
            with httpx.stream("POST", self._base_url, json=payload, timeout=self._timeout) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            
                            # Check if done
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            yield f"\n\nError: {str(e)}"
    
    @property
    def model_name(self) -> str:
        return self._model_name


class OpenAILLM(LLMService):
    """
    OpenAI LLM service (for future use).
    Supports GPT-3.5, GPT-4, etc.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-3.5-turbo", 
        api_key: str = None
    ):
        self._model_name = model_name
        self._api_key = api_key
        
        print(f"✓ Configured OpenAI LLM: {self._model_name}")
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Iterator[str]:
        """Generate using OpenAI API."""
        # TODO: Implement OpenAI API call
        raise NotImplementedError("OpenAI LLM not yet implemented")
    
    @property
    def model_name(self) -> str:
        return self._model_name


# Global LLM service instance
_llm_service: LLMService = None


def get_llm_service() -> LLMService:
    """
    Get or create global LLM service instance.
    Singleton pattern.
    """
    global _llm_service
    
    if _llm_service is None:
        provider = settings.LLM_PROVIDER.lower()
        
        if provider == "ollama":
            _llm_service = OllamaLLM()
        elif provider == "openai":
            _llm_service = OpenAILLM()
        else:
            # Default to Ollama
            _llm_service = OllamaLLM()
    
    return _llm_service


def generate_answer(
    prompt: str, 
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False
) -> str | Iterator[str]:
    """
    Convenience function to generate text.
    
    Args:
        prompt: Input prompt
        max_tokens: Max tokens
        temperature: Temperature
        stream: Stream response
    
    Returns:
        Generated text or iterator
    """
    service = get_llm_service()
    return service.generate(prompt, max_tokens, temperature, stream)


def build_rag_prompt(query: str, context_chunks: list[Dict[str, Any]]) -> str:
    """
    Build RAG prompt with context and query.
    
    Args:
        query: User query
        context_chunks: List of relevant chunks with metadata
    
    Returns:
        Formatted prompt string
    """
    # Build context section
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", "")
        context_parts.append(f"{text}\n")
    
    context_text = "\n".join(context_parts)
    
    # Build full prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from documents.

Context from documents:
{context_text}

User Question: {query}

Instructions:
- Answer using ONLY the information from the context above
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Do NOT mention source numbers, page numbers, or document references in your answer
- Do not make up information or use knowledge outside the context
- Provide a natural, flowing answer without citations

Answer:"""
    
    return prompt
