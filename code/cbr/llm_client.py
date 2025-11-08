"""
LLM Client for PSS Teaching Personas
Supports OpenAI GPT and Anthropic Claude models
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 500
    api_key: Optional[str] = None


class LLMClient:
    """
    Unified interface for different LLM providers
    
    Supports:
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude 3)
    
    Usage:
        llm = LLMClient(provider="openai", model="gpt-3.5-turbo")
        response = llm.generate_teaching_response(
            misconception="adds numerators and denominators",
            strategy="socratic"
        )
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: Optional[str] = None
    ):
        self.config = LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        
        self._initialize_client()
        logger.info(f"Initialized {provider} LLM client with model {model}")
    
    def _initialize_client(self):
        """Initialize provider-specific client"""
        provider = self.config.provider.lower()
        
        if provider == "openai":
            self._init_openai()
        elif provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Run: pip install openai"
            )
        
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = openai.OpenAI(api_key=api_key)
        logger.debug("OpenAI client initialized")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic library not installed. Run: pip install anthropic"
            )
        
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.debug("Anthropic client initialized")
    
    def generate_teaching_response(
        self,
        misconception: str,
        context: str = "mathematics",
        strategy: str = "traditional_teaching",
        retrieved_cases: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate teaching response using LLM
        
        Args:
            misconception: Description of student misconception
            context: Subject/topic context
            strategy: Teaching strategy to use
            retrieved_cases: Similar historical cases (for hybrid system)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
        
        Returns:
            Generated teaching response text
        """
        # Build prompt
        prompt = self._build_prompt(
            misconception=misconception,
            context=context,
            strategy=strategy,
            retrieved_cases=retrieved_cases
        )
        
        # Get system prompt
        system_prompt = self._get_system_prompt(strategy)
        
        # Use provided overrides or defaults
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens
        
        # Generate response
        try:
            if self.config.provider == "openai":
                response = self._generate_openai(prompt, system_prompt, temp, max_tok)
            elif self.config.provider == "anthropic":
                response = self._generate_anthropic(prompt, system_prompt, temp, max_tok)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            logger.debug(f"Generated response ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._fallback_response(misconception, strategy)
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using Anthropic API"""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _get_system_prompt(self, strategy: str) -> str:
        """
        Get system prompt for specific teaching strategy
        
        Based on established pedagogical approaches:
        - Socratic: Questioning and discovery
        - Constructive: Scaffolding and ZPD
        - Experiential: Real-world contexts
        - Rule-based: Direct instruction
        - Traditional: Classical teaching
        """
        prompts = {
            'socratic': """You are a Socratic mathematics tutor. Your teaching approach:

1. NEVER give direct answers - always guide through questions
2. Ask probing questions that expose contradictions in student reasoning
3. Build on student responses with follow-up questions
4. Help students discover correct understanding themselves
5. Use counterexamples to highlight flaws in reasoning

Example:
Student: "I add 1/3 + 1/4 = 2/7"
You: "Interesting approach. What happens if we try your method with 1/2 + 1/2?"

Keep responses concise (2-3 sentences) and focused on questioning.""",

            'constructive': """You are a constructive mathematics tutor. Your teaching approach:

1. Build on what the student already knows
2. Provide scaffolding appropriate to their current level (Zone of Proximal Development)
3. Break complex problems into manageable steps
4. Gradually increase difficulty as student demonstrates mastery
5. Offer hints and support without giving away answers

Example:
Student: "I don't understand fractions"
You: "Let's start with what you DO know. Can you divide a pizza into equal slices? That's actually fractions! Now let's connect that to 1/4..."

Adjust your support level dynamically based on student progress.""",

            'experiential': """You are an experiential mathematics tutor. Your teaching approach:

1. Connect abstract math to real-world situations
2. Use concrete analogies and examples
3. Draw from everyday experiences students can relate to
4. Make mathematical relationships tangible and visual
5. Show practical applications

Example:
Student: "Why do we need common denominators?"
You: "Imagine comparing two recipes - one needs 1/3 cup flour, another needs 1/4 cup. To compare them, you'd want the same measuring cup size, right? That's exactly what common denominators do!"

Always ground explanations in familiar contexts.""",

            'rule_based': """You are a rule-based mathematics tutor. Your teaching approach:

1. Provide explicit step-by-step procedures
2. State rules clearly and directly
3. Show worked examples with each step labeled
4. Give immediate corrective feedback
5. Focus on correct algorithmic execution

Example:
Student: "I add 1/3 + 1/4 = 2/7"
You: "That's incorrect. Here's the proper procedure:
Step 1: Find common denominator (LCD of 3 and 4 = 12)
Step 2: Convert: 1/3 = 4/12, 1/4 = 3/12
Step 3: Add numerators: 4/12 + 3/12 = 7/12
Always find the LCD first before adding fractions."

Be direct and procedural in all explanations.""",

            'traditional_teaching': """You are a traditional mathematics teacher. Your teaching approach:

1. Explain concepts clearly and thoroughly
2. Provide worked examples
3. Check for understanding
4. Assign practice problems
5. Give straightforward corrections when needed

Example:
Student: "I don't understand fraction addition"
You: "Let me explain. When adding fractions, you need the same denominator - the bottom number. If they're different, find a common denominator by finding a number both bottom numbers divide into. Then convert each fraction and add the top numbers. Let's try 1/3 + 1/4 together..."

Balance clarity with comprehensiveness."""
        }
        
        return prompts.get(strategy, prompts['traditional_teaching'])
    
    def _build_prompt(
        self,
        misconception: str,
        context: str,
        strategy: str,
        retrieved_cases: Optional[List[Dict]] = None
    ) -> str:
        """Build prompt with context and optional case grounding"""
        
        prompt = f"""Student Misconception: {misconception}

Subject: {context}

Teaching Strategy: {strategy}
"""
        
        # Add retrieved cases for grounding (hybrid system)
        if retrieved_cases:
            prompt += "\n\nSimilar Historical Cases:\n"
            for i, case in enumerate(retrieved_cases[:3], 1):
                misconception_text = case.get('misconception', 'Unknown')
                success_rate = case.get('success_rate', 0.0)
                prompt += f"{i}. Misconception: {misconception_text}\n"
                prompt += f"   Previous success rate: {success_rate:.2f}\n"
            
            prompt += "\nBased on these successful past interventions, "
        
        prompt += "generate an appropriate teaching response:"
        
        return prompt
    
    def _fallback_response(self, misconception: str, strategy: str) -> str:
        """Fallback response when LLM fails"""
        fallbacks = {
            'socratic': f"What makes you think that approach works? Can you test it with a simpler example?",
            'constructive': f"Let's break this down. What do you already know about this topic?",
            'experiential': f"Think of a real-world situation where this comes up. How would you handle it there?",
            'rule_based': f"Let me show you the correct procedure step by step.",
            'traditional_teaching': f"Let me explain the correct way to approach {misconception}."
        }
        return fallbacks.get(strategy, "Let's work through this together.")
    
    def estimate_cost(self, n_calls: int, avg_tokens_per_call: int = 500) -> float:
        """
        Estimate cost for n API calls
        
        Args:
            n_calls: Number of API calls
            avg_tokens_per_call: Average tokens per call (prompt + completion)
        
        Returns:
            Estimated cost in USD
        """
        total_tokens = n_calls * avg_tokens_per_call
        
        # Cost per 1M tokens (as of 2024)
        costs = {
            "gpt-3.5-turbo": 2.00,
            "gpt-4-turbo": 30.00,
            "gpt-4": 60.00,
            "claude-3-haiku-20240307": 1.00,
            "claude-3-sonnet-20240229": 15.00,
            "claude-3-opus-20240229": 75.00
        }
        
        cost_per_million = costs.get(self.config.model, 5.00)  # Default estimate
        estimated_cost = (total_tokens / 1_000_000) * cost_per_million
        
        return estimated_cost
    
    def __repr__(self) -> str:
        return (f"LLMClient(provider={self.config.provider}, "
                f"model={self.config.model})")


# Convenience functions
def create_openai_client(model: str = "gpt-3.5-turbo", **kwargs) -> LLMClient:
    """Create OpenAI LLM client"""
    return LLMClient(provider="openai", model=model, **kwargs)


def create_anthropic_client(model: str = "claude-3-haiku-20240307", **kwargs) -> LLMClient:
    """Create Anthropic LLM client"""
    return LLMClient(provider="anthropic", model=model, **kwargs)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: OpenAI
    print("Example 1: OpenAI GPT-3.5-turbo")
    print("-" * 50)
    try:
        llm = create_openai_client()
        response = llm.generate_teaching_response(
            misconception="Student adds numerators and denominators when adding fractions",
            strategy="socratic"
        )
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Anthropic Claude
    print("Example 2: Anthropic Claude")
    print("-" * 50)
    try:
        llm = create_anthropic_client()
        response = llm.generate_teaching_response(
            misconception="Student adds numerators and denominators when adding fractions",
            strategy="experiential",
            retrieved_cases=[
                {
                    'misconception': 'fraction addition',
                    'success_rate': 0.85
                }
            ]
        )
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Cost estimation
    print("Example 3: Cost Estimation")
    print("-" * 50)
    llm = create_openai_client()
    cost = llm.estimate_cost(n_calls=1000, avg_tokens_per_call=500)
    print(f"Estimated cost for 1000 calls: ${cost:.2f}")
