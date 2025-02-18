from typing import Optional, List
from enum import Enum
import logging
import openai
from dataclasses import dataclass
import base64


class ModelType(Enum):
    """Supported model types"""
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
    GPT4V = "gpt-4o"


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
            if self.config.model_type in [ModelType.GPT4, ModelType.GPT35, ModelType.GPT4V]:
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

    
    def generate(self, 
                system_prompt: str, 
                user_prompt: str, 
                image_paths: Optional[List[str]] = None,
                temperature: Optional[float] = None) -> str:
        """
        Generate text using the configured model
        
        Args:
            system_prompt: System context/instruction
            user_prompt: User input/query
            image_paths: Optional list of paths to images for vision tasks
            temperature: Optional temperature override
            
        Returns:
            str: Generated text response
        """
        temp = temperature if temperature is not None else self.config.temperature
        
        try:
            if self.config.model_type == ModelType.GPT4V:
                if not image_paths:
                    raise ValueError(f"Image paths required for {ModelType.GPT4V} model")
                return self._generate_openai_vision(system_prompt, user_prompt, image_paths, temp)
            elif self.config.model_type in [ModelType.GPT4, ModelType.GPT35]:
                return self._generate_openai(system_prompt, user_prompt, temp)
            else:
                self.logger.error(f"Unsupported model type")
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
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
        
        messages.append({"role": "user", "content": content})
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=self.config.max_length
        )
        
        return response.choices[0].message.content

    
    
def main():
    ######################################
    ######### Generate text Only ########
    ######################################
    # gpt4_config = ModelConfig(
    #     model_type=ModelType.GPT4,
    #     api_key="API-key"
    # )
    # gpt4_client = LLMClient(config=gpt4_config)
    
    # system_prompt = "You are an outstanding clothing stylist. You are good at matching and designing eye-catching looks and keeping people at an appropriate temperature and comfort level."
    # user_prompt = "These are the clothes I need for dressing up within a week. The average temperature this week is between 10 and 20 degrees Celsius. There are no special activities this week, and the main activity is going to work. Please help me match the clothes according to my pictures, the temperature and the scene."
    
    # response = gpt4_client.generate(system_prompt, user_prompt)
    # print("========")
    # print(response)

    ######################################
    ###### Generate text with images #####
    ######################################
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



if __name__ == "__main__":
    main()
 
  