"""
LLM Client Interface
Interface for different LLM providers with caching support
"""

import os
import json
from typing import Optional, Dict, Any
from enum import Enum


class Provider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    MOCK = "mock"


class LLMClient:
    """Unified interface for different LLMs with GPT-5 API."""
    
    DEFAULT_MODELS = {
        Provider.OPENAI: "gpt-4o-mini",
        Provider.MOCK: "mock"
    }
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider name
            model: Model identifier
            api_key: API key (uses env var if not provided)
        """
        self.provider = Provider(provider)
        self.model = model or self.DEFAULT_MODELS[self.provider]
        self.api_key = api_key
        self.client = None
        self.is_gpt5 = self.model.startswith("gpt-5") and not self.model.startswith("gpt-5-c")
        
        # Store conversation history for GPT-5 CoT passing
        self.conversation_history = []

        self._setup_client()
    
    def _setup_client(self) -> None:
        """Initialize the appropriate API client."""
        if self.provider == Provider.OPENAI:
            self._setup_openai()
        # MOCK provider needs no setup
    
    def _setup_openai(self) -> None:
        """Setup OpenAI client."""
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            print(" OpenAI not installed. Install with: pip install openai")
            self.provider = Provider.MOCK
            return
        
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(" No OpenAI API key found. Using mock responses.")
            self.provider = Provider.MOCK
            return
        
        self.client = OpenAI(api_key=api_key)
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        reasoning_effort: str = "high",  # New parameter for GPT-5
        verbosity: str = "medium",  # New parameter for GPT-5
        use_chain_of_thought: bool = True  # Whether to use CoT for GPT-5
    ) -> str:
        """
        Call LLM with the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Response randomness (0=deterministic)
            max_tokens: Maximum response length
            reasoning_effort: GPT-5 reasoning effort (minimal/standard/high)
            verbosity: GPT-5 response verbosity (low/medium/high)
            use_chain_of_thought: Whether to pass CoT between turns for GPT-5
            
        Returns:
            Model response text
        """
        if self.provider == Provider.MOCK:
            return self._mock_response(prompt)
        
        try:
            if self.provider == Provider.OPENAI:
                if self.is_gpt5:
                    return self._call_gpt5_responses_api(
                        prompt, system_prompt, temperature, max_tokens,
                        reasoning_effort, verbosity, use_chain_of_thought
                    )
                else:
                    return self._call_openai(
                        prompt, system_prompt, temperature, max_tokens
                    )
        except ImportError as e:
            print(f" Import Error: {e}")
            print("  Falling back to mock response")
            return self._mock_response(prompt)
        except Exception as e:
            print(f" API Error: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()[:500]}")
            print("  Falling back to mock response")
            return self._mock_response(prompt)
    
    def _call_gpt5_responses_api(
    self,
    prompt: str,
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    verbosity: str,
    use_chain_of_thought: bool
    ) -> str:
        """Call GPT-5 using the new Responses API."""
        try:
            import requests
        except ImportError:
            raise Exception("requests library not installed")
        
        # Get API key
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("No OpenAI API key found")
        
        # Build the input
        if system_prompt:
            input_text = f"{system_prompt}\n\n{prompt}"
        else:
            input_text = prompt
        
        # Build request payload - SIMPLE VERSION WITHOUT CONVERSATION HISTORY
        # The conversation history is causing issues with reasoning/message pairing
        payload = {
            "model": self.model,
            "input": input_text,
            "reasoning": {
                "effort": reasoning_effort
            },
            "text": {
                "verbosity": verbosity
            },
            "max_output_tokens": max_tokens
        }
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=payload,
                timeout=600  # Increased timeout for GPT-5
            )
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
        
        if response.status_code != 200:
            raise Exception(f"GPT-5 API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Check if response is incomplete
        if result.get("status") == "incomplete":
            reason = result.get("incomplete_details", {}).get("reason", "unknown")
            if reason == "max_output_tokens":
                # Still try to extract partial response
                pass
        
        # Extract text from response 
        response_text = ""
        
        # Try the SDK helper first
        if "output_text" in result:
            response_text = result["output_text"]
        
        # If not available, parse output items
        if not response_text:
            output_items = result.get("output", [])
            for item in output_items:
                if item.get("type") == "message":
                    # Look for content
                    content_items = item.get("content", [])
                    if isinstance(content_items, list):
                        for content in content_items:
                            if isinstance(content, dict):
                                # Try different content type fields
                                if content.get("type") == "output_text":
                                    response_text = content.get("text", "")
                                    if response_text:
                                        break
                                elif content.get("type") == "text":
                                    response_text = content.get("text", "")
                                    if response_text:
                                        break
                            elif isinstance(content, str):
                                # Sometimes content is just a string
                                response_text = content
                                break
                    elif isinstance(content_items, str):
                        # Sometimes content is directly a string
                        response_text = content_items
                    
                    # Also check for direct text field on message
                    if not response_text and "text" in item:
                        response_text = item["text"]
                    
                    if response_text:
                        break
        
        # Last resort - return the raw JSON
        if not response_text:
            response_text = json.dumps(result, indent=2)
        
        # Clear conversation history since we're not using it
        self.conversation_history = []
        
        return response_text
    
    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call OpenAI API using Chat Completions (for non-GPT-5 models)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history (for GPT-5 CoT)."""
        self.conversation_history = []
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        prompt_lower = prompt.lower()
        
        if "circular" in prompt_lower or "lbo" in prompt_lower:
            return """Let me calculate the LBO model:
            
Initial values:
- Equity = $400M
- Debt = Total_Sources × 0.6
- Fees = Debt × 0.02  
- Total_Sources = Equity + Debt + Fees

This creates a circular reference. Using iterative approach:

Iteration 1:
- Initial Total_Sources = $400M (equity only)
- Debt = $400M × 0.6 = $240M
- Fees = $240M × 0.02 = $4.8M
- New Total_Sources = $400M + $240M + $4.8M = $644.8M

Iteration 2:
- Debt = $644.8M × 0.6 = $386.88M
- Fees = $386.88M × 0.02 = $7.74M
- Total_Sources = $400M + $386.88M + $7.74M = $794.62M

[Converged after 5 iterations]
Final values:
- Total_Sources = $1,010.20M
- Debt = $606.12M
- Fees = $12.12M
- Equity = $400M"""
        
        return """Step-by-step calculation:

Given:
- Revenue = $100M
- Costs = $60M
- Tax_Rate = 25%

Calculations:
- EBITDA = Revenue - Costs = $100M - $60M = $40M
- Tax = EBITDA × Tax_Rate = $40M × 0.25 = $10M
- Net_Income = EBITDA - Tax = $40M - $10M = $30M

Dependencies tracked:
- EBITDA depends on: Revenue, Costs
- Net_Income depends on: EBITDA, Tax_Rate
- If Revenue changes, must recalculate: EBITDA, Net_Income"""