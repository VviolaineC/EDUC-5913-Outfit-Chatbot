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

