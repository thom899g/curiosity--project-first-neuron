"""
Unified LLM Client - The core abstraction enabling seamless transplantation between cloud and local inference.
Architectural Principle: One system, two deployment states.
"""
import os
import json
import logging
from enum import Enum
from typing import Literal, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp
import backoff
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMBackend(str, Enum):
    """Supported LLM backends"""
    OPENAI = "openai"
    LOCAL_OLLAMA = "local_ollama"
    LOCAL_LM_STUDIO = "local_lm_studio"


class LLMResponse(BaseModel):
    """Structured LLM response"""
    content: str
    reasoning: str = Field(default="")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = Field(default=0)
    backend: LLMBackend


@dataclass
class LLMConfig:
    """LLM configuration"""
    backend: LLMBackend
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30


class LLMClient:
    """Unified LLM client with automatic backend switching"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration and environment"""
        if self.config.backend == LLMBackend.OPENAI:
            if not self.config.api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable required for OpenAI backend")
                self.config.api_key = api_key
            if not self.config.base_url:
                self.config.base_url = "https://api.openai.com/v1"
                
        elif self.config.backend == LLMBackend.LOCAL_OLLAMA:
            self.config.base_url = self.config.base_url or "http://localhost:11434"
            self.config.api_key = None  # Ollama doesn't require API key
            
        elif self.config.backend == LLMBackend.LOCAL_LM_STUDIO:
            self.config.base_url = self.config.base_url or "http://localhost:1234/v1"
            # LM Studio uses OpenAI-compatible API but without key
            
    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists"""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, TimeoutError),
        max_tries=3,
        logger=logger
    )
    async def _call_openai(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Call OpenAI API with exponential backoff"""
        await self._ensure_session()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            async with self._session.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                
                # Parse structured response if possible
                try:
                    parsed = json.loads(content)
                    reasoning = parsed.get("reasoning", "")
                    confidence = float(parsed.get("confidence", 0.5))
                    content = parsed.get("content", content)
                    metadata = parsed.get("metadata", {})
                except json.JSONDecodeError:
                    reasoning = ""
                    confidence = 0.5
                    metadata = {}
                
                return LLMResponse(
                    content=content,
                    reasoning=reasoning,
                    confidence=confidence,
                    metadata=metadata,
                    tokens_used=tokens_used,
                    backend=self.config.backend
                )
                
        except aiohttp.ClientResponseError as e:
            logger.error(f"OpenAI API error {e.status}: {e.message}")
            raise
        except KeyError as e:
            logger.error(f"Malformed OpenAI response: {e}")
            raise
            
    async def _call_local_ollama(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """Call local Ollama API"""
        await self._ensure_session()
        
        # Ollama format: system prompt in payload, not in messages array
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        try:
            async with self._session.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                content = data.get("response", "")
                tokens_used = data.get("eval_count", 0)
                
                # Attempt to parse structured response
                try:
                    parsed = json.loads(content)
                    reasoning = parsed.get("reasoning", "")
                    confidence = float(parsed.get("confidence", 0.5))
                    content = parsed.get("content", content)
                    metadata = parsed.get("metadata", {})
                except json.JSONDecodeError:
                    reasoning = ""
                    confidence = 0.5
                    metadata = {}
                
                return LLMResponse(
                    content=content,
                    reasoning=reasoning,
                    confidence=confidence,
                    metadata=metadata,
                    tokens_used=tokens_used,
                    backend=self.config.backend
                )
                
        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            raise
            
    async def reason(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """
        Unified reasoning interface
        Args:
            prompt: User prompt
            system_prompt: System instructions
        Returns:
            Structured LLMResponse
        """
        logger.info(f"LLM reasoning request with backend {self.config.backend}")
        
        if self.config.backend == LLMBackend.OPENAI:
            return await self._call_openai(prompt, system_prompt)
        elif self.config.backend == LLMBackend.LOCAL_OLLAMA:
            return await self._call_local_ollama(prompt, system_prompt)
        elif self.config.backend == LLMBackend.LOCAL_LM_STUDIO:
            # LM Studio uses OpenAI-compatible API
            # Temporarily use OpenAI method with modified config
            original_backend = self.config.backend
            self.config.backend = LLMBackend.OPENAI
            try:
                return await self._call_openai(prompt, system_prompt)
            finally:
                self.config.backend = original_backend
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
            
    async def close(self) -> None:
        """Clean up resources"""
        if self._session and not self._session.closed:
            await self._session.close()
            
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()