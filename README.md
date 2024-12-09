# EDUC-5913-Outfit-Chatbot
:)

## Introduction

💖Hello everyone. The project of Haipei and I wants to establish a vertical chatbot system integrating multiple models for daily clothing matching. We combine text language models and image recognition capabilities to provide real-time feedback based on user-input images, temperatures within a limited time range, and dressing scenarios.

💖大家好，我和海培的项目想要建立一个集成多种模型的垂直领域聊天机器人系统，用于日常服装搭配。我们结合了文本语言模型以及图片识别能力，依据用户输入的图片、限定时间范围内的气温、穿搭所需场景来进行实时反馈。 

😇Since there are two of us writing code simultaneously, maintenance and collaborative construction are of great significance in code editing. For example, every time we add a new model type, we need to make manual modifications in multiple parts of the code, which makes it easy to miss something or introduce errors. 

😇因为我们是两个人同时写作，所以维护和共建在代码编辑中显得十分重要。譬如说， 当我们每次新增一个模型类型，都需要在代码的多处手动修改，容易遗漏或引入错误。

## Potential issues
### 🤯Potential Scenario 1 潜在场景1 
```python
# new additional model "gpt5"
if model_type == "gpt5":
print("Running GPT-5")
```
Issues 问题所在:

• If these strings are used in dozens of places, we need to check one by one to see if they have all been updated.

• Without centralized management, it's very easy to miss something or cause logical conflicts. 

•	如果有几十处地方使用了这些字符串，我们需要逐一检查是否都更新了。

•	没有集中管理时，很容易遗漏或导致逻辑冲突。

### 🤯Potential Scenario 2 潜在场景2

When I want to deploy a local open-source model from Hugging Face by myself, my team doesn't recognize this small self-owned model. Using strings directly can't provide any context information, and at this time, the meaning of the code isn't clear enough. 

当我想要自己部署一个hugging face 上的本地开源模型， 我的team 不认识这个自有小模型， 直接使用字符串无法提供任何上下文信息，此时代码的含义不够清晰。

```python
if model_type == "llama3":
print("Model is LLaMA-3")
```

Issues 问题所在:

•	What is "llama3"? Is it a variable? Or a specific value?

•	"llama3" 是什么？一个变量？还是一个具体的值？

**⚠️Therefore, during the process of writing code, what we consider is not just the construction of the system. How to make our existing system more semantic (clear), safer and easier to maintain is of crucial importance. Hence, in the early stage of building the system, we introduced "Enum". Although the following piece of code is simple, its advantages will become more and more obvious as the program becomes more complex.**

**⚠️因此，在写代码的过程中， 我们考虑的不仅仅是系统的构建， 如何将我们现有系统构建得更语义化（清晰）、更安全、更易维护，至关重要。 所以，在构建系统的前期，我们引入了“Enum”。 
以下段代码虽然简单，但它的好处会在程序变复杂时越来越明显。**


```python
class ModelType(Enum):
    """Supported model types"""
    LLAMA3 = "llama3"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
```
## So, What is the Benefit of Using Enum?

🧠Define a set of fixed model types, which represent a set of options that won't change randomly.

🧠定义一组固定的模型类型，这些模型类型代表了一组不会随意变化的选项。

🤙🏻Improve the readability of code

🤙🏻提高代码的可读性

👌🏻Improve code security

👌🏻提高代码安全性

Besides the issue of string spelling, when using enums, IDEs or code checking tools will automatically prompt errors.

除了字符串拼写的问题， 用枚举时，IDE 或代码检查工具会自动提示错误
```python
if model_type == ModelType.GPT5:  # If GPT5 is not defined, an error will be reported here.
print("This is GPT-4")
```

😁More convenient for expansion and maintenance.

😁更方便的扩展和维护

Suppose a new model needs to be added in the future, such as "gpt5". You can directly add a new member to the enumeration class: 

假设以后需要添加一个新的模型，比如 "gpt5"，可以直接在枚举类中新增成员：

```python
class ModelType(Enum):
    LLAMA3 = "llama3"
    GPT4 = "gpt4"
    GPT35 = "gpt3.5"
GPT5 = "gpt5"  # newly added
```

## Scaling 

😆Similarly, the @dataclass we use later is also considered for the same reason: We hope it can simplify class definitions in the code and enhance its maintainability. 

😆同理，我们在之后使用的@dataclass也是拥有同样的考虑： 希望代码简化类定义、增强其可维护性。

For example, without @dataclass, we would need to write these methods manually:

例如，如果没有 @dataclass，我们需要手动编写这些方法：

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

**但有了@dataclass 之后， 效果完全相同，代码相当简洁。**

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_type: str
    max_length: int = 4096
temperature: float = 0.7
```

## ModelConfig & LLMClient

Since we expect this system to integrate multiple models, we use a unified interface to initialize, manage and utilize various large language models (LLM). Here, we use ModelConfig to store the parameters for model initialization, and the LLMClient is employed to read the content of ModelConfig to load and use the models. 

由于我们希望本系统集成多种模型， 因此我们通过一个统一的接口来初始化、管理和使用各种大语言模型（LLM）。在这里，我们使用使用ModelConfig进行模型初始化的参数存储、 用LLMClient读取 ModelConfig 的内容来加载和使用模型。

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

When we first implemented model initialization, we ran into a couple of issues:

1.	No strict validation
2.	Risk of crashes

### Why Non-Modular Design Doesn’t Work!

If you don’t use a modular approach, you end up putting all the logic for selecting and initializing models directly in the main flow. Here’s why that’s a bad idea:

1.	Redundant code

2.	Hard to maintain

3.	Scalability issues

### Non-Modular Example

Here’s what it looked like without modular design:

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

To fix this, we switched to a modular design. Here’s how it works:


1.	Validate model_type:

o	If it’s GPT4 or GPT3.5, we call _initialize_openai to handle OpenAI models.

o	If it’s LLAMA3, we call _initialize_llama to handle the Llama model.

o	If it’s unsupported, we raise an exception and log the error.

2.	Log the result: We log what happened whether it’s a success or failure.








## About Image upload 关于上传图像

Next, we defined a method named _encode_image, whose function is to convert an image file into a Base64-encoded string. We send the corresponding image data through the API interface. The specific process is as follows: The user uploads an image → The image is encoded into a Base64 string and embedded in the JSON request → The server receives the request and decodes the Base64 string, converting it into a format that can be processed by the Convolutional Neural Network (CNN). 

接着，我们定义了一个方法 _encode_image，它的作用是 将图像文件转换为 Base64 编码的字符串。 我们通过API 接口发送相对应的图像数据。具体路径为：用户上传图片👉图片被编码为 Base64 字符串，嵌入到 JSON 请求中👉服务端接收请求并解码 Base64 字符串，将其转换为 CNN 可处理的格式。

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

通过 GPT-4V 模型实现一个多模态任务，结合文本提示和图像输入，为用户提供基于温度和场景的服装搭配建议。

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
