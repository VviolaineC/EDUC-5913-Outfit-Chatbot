from typing import Optional
from enum import Enum
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
from dataclasses import dataclass


class ModelType(Enum):
    """Supported model types"""
    LLAMA3 = "llama3"
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
            elif self.config.model_type == ModelType.LLAMA3:
                self._initialize_llama()
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
                return self._generate_local(system_prompt, user_prompt, temp)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    def _generate_openai(self, 
                        system_prompt: str, 
                        user_prompt: str, 
                        temperature: float) -> str:
        """Generate text using OpenAI API"""
        model_name = "gpt-4" if self.config.model_type == ModelType.GPT4 else "gpt-3.5-turbo"
        
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=self.config.max_length
        )
        
        return response.choices[0].message.content

    def _initialize_llama(self):
        """Initialize Llama model with proper token configuration"""
        try:
            # Initialize tokenizer with proper defaults
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                padding_side="left"  # Important for attention mask alignment
            )
            
            # Ensure pad token is set correctly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            # Initialize model with safe defaults
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                use_safetensors=True,  # Use safetensors to avoid weight issues
                low_cpu_mem_usage=True  # Help with memory management
            )
            
            # Ensure model and tokenizer vocab sizes match
            if len(self.tokenizer) != self.model.config.vocab_size:
                logging.warning(f"Tokenizer vocab size ({len(self.tokenizer)}) != Model vocab size ({self.model.config.vocab_size})")
                # Resize model embeddings to match tokenizer
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            logging.info(f"Model initialized successfully with vocab size {len(self.tokenizer)}")
            
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise

    def _generate_local(self, 
                   system_prompt: str, 
                   user_prompt: str, 
                   temperature: float = 0.7,
                   max_new_tokens: Optional[int] = None) -> str:
        """Generate text using local models with enhanced handling"""
        try:
            # Format prompt according to Llama 3 chat template
            chat_template = f"""<s>[INST] <<SYS>>{system_prompt}<</SYS>>
    {user_prompt}[/INST]
    """
            # Tokenize with proper handling
            inputs = self.tokenizer(
                chat_template,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True,
            )
            
            # Move inputs to correct device
            input_ids = inputs['input_ids'].to(self.config.device)
            attention_mask = inputs['attention_mask'].to(self.config.device)

            # Set up generation config
            gen_config = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_new_tokens': max_new_tokens or 1024,
                'do_sample': temperature > 0,
                'temperature': temperature,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'use_cache': True
            }

            # Generate with safe defaults
            with torch.no_grad():
                outputs = self.model.generate(**gen_config)

            # Get only the new tokens (exclude input prompt)
            new_tokens = outputs[0][len(input_ids[0]):]
            
            # Decode response
            response = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Clean up any remaining special tokens or formatting
            response = response.replace('[/INST]', '').strip()
            response = ' '.join(response.split())  # Normalize whitespace

            return response

        except Exception as e:
            logging.error(f"Error in text generation: {e}", exc_info=True)
            raise



def main():
    Example 1: Using GPT-4
    gpt4_config = ModelConfig(
        model_type=ModelType.GPT4,
        api_key="your-api-key"
    )
    gpt4_client = LLMClient(config=gpt4_config)
    
    # Example 2: Using Llama
    #llama_config = ModelConfig(
    #    model_type=ModelType.LLAMA3,
    #    model_path="meta-llama/Meta-Llama-3-8b"
    #)
    #llama_client = LLMClient(config=llama_config)
    
    # Generate text
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "What is the capital of France?"
    
    response = llama_client.generate(system_prompt, user_prompt)
    print("========")
    print(response)

if __name__ == "__main__":
    main()
