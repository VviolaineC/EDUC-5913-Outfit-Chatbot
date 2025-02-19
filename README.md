# EDUC5913-Outfit-Chatbot-Coding

## Introduction

This project presents an innovative vertical chatbot system that integrates multiple AI models to address the specific domain of daily clothing coordination. By combining large language models (LLMs) with computer vision capabilities, our system offers contextually aware fashion recommendations based on real-time user inputs.

The proposed system uniquely synthesizes multiple data modalities: visual input through user-submitted clothing images, environmental parameters such as temperature forecasts, and contextual information about specific wearing scenarios. This multi-modal approach enables the system to provide personalized fashion advice that considers both aesthetic and practical factors within a constrained temporal framework.

## Background

While APIs excel at specific tasks, a multimodel agent integrates multiple specialized models (NLP, computer vision, speech processing) to handle complex multi-modal tasks more effectively, enabling comprehensive analysis of varied inputs like text and images.

Cost efficiency is achieved by using lightweight local models for routine tasks while reserving API calls for complex operations. This hybrid approach not only reduces expenses but also enhances data privacy by processing sensitive information locally. 

Multimodel agents provide greater reliability and flexibility since they're not dependent on a single provider's API, eliminating vulnerabilities to external changes or outages. The architecture supports scalability through easy integration of new models and customization of workflows.

This approach empowers developers with deeper control over model selection, fine-tuning, and orchestration, fostering technical innovation. The result is a robust, adaptable system that efficiently handles sophisticated AI tasks requiring multi-modal processing, making it superior to standalone API usage for complex applications.

##  Setting up the Foundational Components

```python
from typing import Optional, List
from enum import Enum
import logging
import openai
from dataclasses import dataclass
import base64
```

## Defining an enumeration (Enum) class named ModelType

This method is used to represent a set of predefined constants corresponding to different model types that the system supports.

```python
class ModelType(Enum):
    """Supported model types"""
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
    GPT4V = "gpt-4o"
```

## Defines a data class ModelConfig
Store configuration settings for initializing and interacting with a specific AI model

```python
@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_type: ModelType
    model_path: Optional[str] = None  # Local path or HuggingFace model ID
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_length: int = 4096
    temperature: float = 0.7
```

## Define a Class LLMClient 

A generic client for interacting with multiple types of language models

```python

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
```

## Private Function _initialize_model

_initialize_model method is a private function within the LLMClient class that initializes the appropriate model based on the configuration provided during the creation of the client.

```python

    def _initialize_model(self):
        """Initialize the appropriate model based on configuration"""
        try:
            if self.config.model_type in [ModelType.GPT4, ModelType.GPT35, ModelType.GPT4V]:
                self._initialize_openai()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
            self.logger.info(f"Successfully initialized {self.config.model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
```

Initializing the OpenAI API Client by Setting up the Necessary Configurations

```python
    def _initialize_openai(self):
        """Initialize OpenAI API client"""
        if not self.config.api_key:
            raise ValueError("API key required for OpenAI models")
            
        openai.api_key = self.config.api_key
        if self.config.api_base:
            openai.api_base = self.config.api_base
    
```

## Encode an Image File into a Base64 String

```python
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

```

## Generate Text Using Configured Model

```python

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
```



## Generate Text Using OpenAI's Language Models

```python
    def _generate_openai_vision(self,
                              system_prompt: str,
                              user_prompt: str,
                              image_paths: List[str],
                              temperature: float) -> str:
        """Generate text using OpenAI GPT-4V API"""
        messages = [{"role": "system", "content": system_prompt}]
        
        content = []
        content.append({"type": "text", "text": user_prompt})
        
        for image_path in image_paths:
            base64_image = self._encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
```

## Generating Text Responses Using both Text-only Input and Text with Image Input
```python
def main():
   
    gpt4v_config = ModelConfig(
        model_type=ModelType.GPT4V,
        api_key="API-key"
    )
    gpt4v_client = LLMClient(config=gpt4v_config)

    
    system_prompt = "You are an outstanding clothing stylist. You are good at matching and designing eye-catching looks and keeping people at an appropriate temperature and comfort level."
    user_prompt = "These are the clothes I need for dressing up within a week. The average temperature this week is between 10 and 20 degrees Celsius. There are no special activities this week, and the main activity is going to work. Please help me match the clothes according to my pictures, the temperature and the scene."
    image_paths = ["imgs/croptop.png", "imgs/hoodie.webp", "imgs/jacket.png", "imgs/shirt.png", "imgs/trenchcoat.png"]
    
    response = gpt4v_client.generate(system_prompt, user_prompt, image_paths)
    print("========")
    print(response)

```






