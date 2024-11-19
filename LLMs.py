from typing import Optional
from enum import Enum
import logging
import openai
from dataclasses import dataclass


class ModelType(Enum):
    """Supported model types"""
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_type: ModelType
    model_path: Optional[str] = None  # Local path or HuggingFace model ID
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_length: int = 4096
    temperature: float = 0.7


class LLMClient:
    """Generic LLM client supporting multiple model types"""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize LLM client with specified configuration
        
        Args:
            config: ModelConfig instance with model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
        # Initialize the specified model type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model based on configuration"""
        try:
            if self.config.model_type in [ModelType.GPT4, ModelType.GPT35]:
                self._initialize_openai()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
            self.logger.info(f"Successfully initialized {self.config.model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def _initialize_openai(self):
        """Initialize OpenAI API client"""
        if not self.config.api_key:
            raise ValueError("API key required for OpenAI models")
            
        openai.api_key = self.config.api_key
        if self.config.api_base:
            openai.api_base = self.config.api_base

    
    def generate(self, 
                system_prompt: str, 
                user_prompt: str, 
                temperature: Optional[float] = None) -> str:
        """
        Generate text using the configured model
        
        Args:
            system_prompt: System context/instruction
            user_prompt: User input/query
            temperature: Optional temperature override
            
        Returns:
            str: Generated text response
        """
        temp = temperature if temperature is not None else self.config.temperature
        
        try:
            if self.config.model_type in [ModelType.GPT4, ModelType.GPT35]:
                return self._generate_openai(system_prompt, user_prompt, temp)
            else:
                self.logger.error(f"model should be gpt3.5 or gpt4")
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    def _generate_openai(self, 
                        system_prompt: str, 
                        user_prompt: str, 
                        temperature: float) -> str:
        """Generate text using OpenAI API"""
        model_name = "gpt-4" if self.config.model_type == ModelType.GPT4 else "gpt-3.5-turbo"
        
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=self.config.max_length
        )
        
        return response.choices[0].message.content

    
    
def main():
    gpt4_config = ModelConfig(
        model_type=ModelType.GPT4,
        api_key="your_API_Key"
    )
    gpt4_client = LLMClient(config=gpt4_config)
    
    
    # Generate text
    system_prompt = "You are an outstanding clothing stylist. You are good at matching and designing eye-catching looks and keeping people at an appropriate temperature and comfort level."
    user_prompt = "What is the best outfit for me, today is 10 degrees Celsius?"
    
    response = gpt4_client.generate(system_prompt, user_prompt)
    print("========")
    print(response)

if __name__ == "__main__":
    main()
