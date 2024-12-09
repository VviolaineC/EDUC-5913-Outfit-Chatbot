# EDUC5913-Outfit-Chatbot-Haipei Li & Zihan Cao
:)

## Introduction

💖Hello everyone. The project of Haipei and Zihan wants to establish a vertical chatbot system integrating multiple models for daily clothing matching. We combine text language models and image recognition capabilities to provide real-time feedback based on user-input images, temperatures within a limited time range, and dressing scenarios.

😇Since there are two of us writing code simultaneously, maintenance and collaborative construction are of great significance in code editing. For example, every time we add a new model type, we need to make manual modifications in multiple parts of the code, which makes it easy to miss something or introduce errors. 

## Potential issues
### 🤯Potential Scenario 1 
```python
# new additional model "gpt5"
if model_type == "gpt5":
print("Running GPT-5")
```
Issues:

• If these strings are used in dozens of places, we need to check one by one to see if they have all been updated.

• Without centralized management, it's very easy to miss something or cause logical conflicts. 


### 🤯Potential Scenario 2 

When I want to deploy a local open-source model from Hugging Face by myself, my team doesn't recognize this small self-owned model. Using strings directly can't provide any context information, and at this time, the meaning of the code isn't clear enough. 


```python
if model_type == "llama3":
print("Model is LLaMA-3")
```

Issues:

•	What is "llama3"? Is it a variable? Or a specific value?



**⚠️Therefore, during the process of writing code, what we consider is not just the construction of the system. How to make our existing system more semantic (clear), safer and easier to maintain is of crucial importance. Hence, in the early stage of building the system, we introduced "Enum". Although the following piece of code is simple, its advantages will become more and more obvious as the program becomes more complex.**



```python
class ModelType(Enum):
    """Supported model types"""
    LLAMA3 = "llama3"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
```
## So, What is the Benefit of Using Enum?

🧠Define a set of fixed model types, which represent a set of options that won't change randomly.


🤙🏻Improve the readability of code



👌🏻Improve code security



Besides the issue of string spelling, when using enums, IDEs or code checking tools will automatically prompt errors.


```python
if model_type == ModelType.GPT5:  # If GPT5 is not defined, an error will be reported here.
print("This is GPT-4")
```

😁More convenient for expansion and maintenance.



Suppose a new model needs to be added in the future, such as "gpt5". You can directly add a new member to the enumeration class: 



```python
class ModelType(Enum):
    LLAMA3 = "llama3"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
GPT5 = "gpt5"  # newly added
```

## Scaling 

😆Similarly, the @dataclass we use later is also considered for the same reason: We hope it can simplify class definitions in the code and enhance its maintainability. 



For example, without @dataclass, we would need to write these methods manually:



```python
class ModelConfig:
    def __init__(self, model_type, max_length=4096, temperature=0.7):
        self.model_type = model_type
        self.max_length = max_length
        self.temperature = temperature

    def __repr__(self):
        return f"ModelConfig(model_type={self.model_type}, max_length={self.max_length}, temperature={self.temperature})"

    def __eq__(self, other):
        if not isinstance(other, ModelConfig):
            return False
        return (self.model_type == other.model_type and
                self.max_length == other.max_length and
                self.temperature == other.temperature)
```

**However, after using @dataclass, the effect is exactly the same and the code is quite concise.**



```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_type: str
    max_length: int = 4096
temperature: float = 0.7
```

## ModelConfig & LLMClient

⛄️ Since we expect this system to integrate multiple models, we use a unified interface to initialize, manage and utilize various large language models (LLM). Here, we use ModelConfig to store the parameters for model initialization, and the LLMClient is employed to read the content of ModelConfig to load and use the models. 



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

## Model Initialization 

### The Problem

🥲 When we first implemented model initialization, we ran into a couple of issues:

1.	No strict validation
2.	Risk of crashes

### Why Non-Modular Design Doesn’t Work!

If you don’t use a modular approach, you end up putting all the logic for selecting and initializing models directly in the main flow. Here’s why that’s a bad idea:

1.	Redundant code ✨

2.	Hard to maintain 🌟

3.	Scalability issues 💫

### Non-Modular Example

🍋 Here’s what it looked like without modular design:

```python
try:
    if config.model_type in [ModelType.GPT4, ModelType.GPT35]:
        if not config.api_key:
            raise ValueError("API key required for OpenAI models")
        openai.api_key = config.api_key
        if config.api_base:
            openai.api_base = config.api_base
        logger.info(f"Successfully initialized OpenAI model: {config.model_type}")
        
    elif config.model_type == ModelType.LLAMA3:
        # Logic for initializing LLAMA3
        logger.info("Successfully initialized LLAMA3 model")
        
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
        
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    raise
```

As you can see, all the initialization logic is crammed into one place. It’s messy, hard to read, and risky when it comes to adding or modifying features.


### The Modular Solution


🪐 To fix this, we switched to a modular design. Here’s how it works:


1.	Validate model_type:

- If it’s GPT4 or GPT3.5, we call _initialize_openai to handle OpenAI models.

- If it’s LLAMA3, we call _initialize_llama to handle the Llama model.

- If it’s unsupported, we raise an exception and log the error.

2.	Log the result: We log what happened whether it’s a success or failure.

### Modular Process

👀 1.	Check if the model type is supported:

- GPT4 / GPT3.5 → Call _initialize_openai.
- LLAMA3 → Call _initialize_llama.

🧠 2.	If unsupported, throw an error and log it.

🕶 3.	Log successful initialization.

## Switched from "print" to "logger"

### The Problem with print
💫 In the beginning, we used print to output debug information, but we quickly realized its limitations:
1.	No categorization
2.	Inflexible
3.	Inefficient debugging

### Why Logger is Better
1.	Real-time monitoring💥

```python
self.logger.info(f"Successfully initialized {self.config.model_type} model")
```

2.	Issue tracking🌟

```python
self.logger.error(f"Error initializing model: {e}")
```

3.	Historical records☀️


### What Logger Brings to the Table

1.	Clear log levels🫧
2.	Better maintainability✨
3.	Faster debugging⚡️

## Summary and Reflection

### Lessons Learned

☀️1. Why logging matters
✨2. The power of modular design

### Where We’re Headed Next
💫1. Expand model initialization
🪐2. Improve logging

## About Image upload

Next, we defined a method named _encode_image, whose function is to convert an image file into a Base64-encoded string. We send the corresponding image data through the API interface. The specific process is as follows: The user uploads an image → The image is encoded into a Base64 string and embedded in the JSON request → The server receives the request and decodes the Base64 string, converting it into a format that can be processed by the Convolutional Neural Network (CNN). 



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

## Prompt engineering

Last but not least, We have defined system prompt, user_prompt, image_paths and temperature.

```python
    def generate(self, 
                system_prompt: str, 
                user_prompt: str, 
                image_paths: Optional[List[str]] = None,
                temperature: Optional[float] = None) -> str:
```

## Outcome!



Implement a multi-modal task through the GPT-4V model, combining text prompts and image inputs to provide users with clothing matching suggestions based on temperature and scenarios. 

```python
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
