# EDUC-5913-Outfit-Chatbot
:)
Hello everyone. The project of Haipei and I wants to establish a vertical chatbot system integrating multiple models for daily clothing matching. We combine text language models and image recognition capabilities to provide real-time feedback based on user-input images, temperatures within a limited time range, and dressing scenarios.

大家好，我和海培的项目想要建立一个集成多种模型的垂直领域聊天机器人系统，用于日常服装搭配。我们结合了文本语言模型以及图片识别能力，依据用户输入的图片、限定时间范围内的气温、穿搭所需场景来进行实时反馈。 
```python
from typing import Optional
from enum import Enum
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
from dataclasses import dataclass
```
Define the name in case of miswritting
```python
class ModelType(Enum):
    """Supported model types"""
    LLAMA3 = "llama3"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
```
Class definition using the Python @dataclass decorator for configuring the initialization parameters of an AI model.
```python
class ModelConfig:
    """Configuration for model initialization"""
    model_type: ModelType
    model_path: Optional[str] = None  # Local path or HuggingFace model ID
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_length: int = 4096
    temperature: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
```
Initialize the appropriate model based on configuration
```python
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

```
Lets generate texts this time
```python
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

```
lets try chatgpt4 this time :)

```python
def main():
    Example 1: Using GPT-4
    gpt4_config = ModelConfig(
        model_type=ModelType.GPT4,
        api_key="your-api-key"
    )
    gpt4_client = LLMClient(config=gpt4_config)
```
define your own output:
```python
# Generate text
    system_prompt = "You are an outstanding clothing stylist. You are good at matching and designing eye-catching looks and keeping people at an appropriate temperature and comfort level."
    user_prompt = "What is the best outfit for me, today is 10 degrees Celsius?"
    
    response = llama_client.generate(system_prompt, user_prompt)
    print("========")
    print(response)

if __name__ == "__main__":
    main()
```

## what to do in the future:
We are planning to integrate the image recognize function within this final project.

## Outcome
![](outcome.png)

