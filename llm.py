"""
LLM Provider Class
Supports multiple LLM providers through LangChain with configurable endpoints
"""

import yaml
import os
from typing import Any, Optional, Dict, Literal
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage



class LLMProvider:
    """
    Universal LLM provider class supporting OpenAI, Google GenAI, Claude, and Azure OpenAI
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize LLM provider with configuration
        
        Args:
            config_path: Path to configuration file (optional if config is provided)
            config: Configuration dictionary (optional if config_path is provided)
            
        Note:
            Either config_path or config must be provided, not both.
            If both are provided, config takes precedence.
        """
        if config is not None:
            self.llm_config = config
        elif config_path is not None:
            self.llm_config = self._load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.llm = self._initialize_llm()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f).get("llm", {})
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration"""
        llm_type = self.llm_config.get('type', '').lower()
        api_key = self.llm_config.get('api_key')
        model = self.llm_config.get('model')
        temperature = self.llm_config.get('temperature', 0)
        base_url = self.llm_config.get('base_url')
        
        if not api_key:
            raise ValueError("API key is required in configuration")
        if not model:
            raise ValueError("Model name is required in configuration")
        
        # Set environment variable for API key (required by some providers)
        self._set_api_key_env(llm_type, api_key)
        
        if llm_type == "openai":
            return self._create_openai_llm(api_key, model, temperature, base_url)
        elif llm_type == "googlegenai":
            return self._create_google_llm(api_key, model, temperature)
        elif llm_type == "claude":
            return self._create_claude_llm(api_key, model, temperature, base_url)
        elif llm_type == "azureopenai":
            return self._create_azure_llm(api_key, model, temperature, base_url)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def _set_api_key_env(self, llm_type: str, api_key: str):
        """Set appropriate environment variable for API key"""
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "googlegenai": "GOOGLE_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "azureopenai": "AZURE_OPENAI_API_KEY"
        }
        
        if llm_type in env_vars:
            os.environ[env_vars[llm_type]] = api_key
    
    def _create_openai_llm(self, api_key: str, model: str, temperature: float, base_url: Optional[str]):
        """Create OpenAI LLM instance"""
        kwargs = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2
        }
        
        if base_url:
            kwargs["base_url"] = base_url
            
        return ChatOpenAI(**kwargs)
    
    def _create_google_llm(self, api_key: str, model: str, temperature: float):
        """Create Google Generative AI LLM instance"""
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    
    def _create_claude_llm(self, api_key: str, model: str, temperature: float, base_url: Optional[str]):
        """Create Claude (Anthropic) LLM instance"""
        kwargs = {
            "model": model,
            "temperature": temperature,
            "anthropic_api_key": api_key,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2
        }
        
        if base_url:
            kwargs["base_url"] = base_url
            
        return ChatAnthropic(**kwargs)
    
    def _create_azure_llm(self, api_key: str, model: str, temperature: float, base_url: Optional[str]):
        """Create Azure OpenAI LLM instance"""
        if not base_url:
            raise ValueError("base_url is required for Azure OpenAI")
        
        # Get API version from config, default to 2025-01-01-preview
        api_version = self.llm_config.get('api_version', '2025-01-01-preview')
            
        return AzureChatOpenAI(
            azure_deployment=model,
            temperature=temperature,
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    
    def call_llm(self, system_prompt: str, user_prompt: str, response_format: Any = None) -> str:
        """
        Call LLM with provided system prompt and user prompt
        
        Args:
            system_prompt: The system prompt to set context
            user_prompt: The user prompt/question
            response_format: Optional structured output format (Pydantic model)
            
        Returns:
            LLM response as string or structured object
        """
        try:
            # Create LLM with structured output if format is specified
            llm = self.llm
            if response_format is not None:
                llm = llm.with_structured_output(response_format)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("user", "{user_prompt}")
            ])
            
            # Create chain
            chain = prompt | llm
            
            self.invoke_model_info('sync')

            # Invoke chain
            response = chain.invoke({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            })
            
            return response
            
        except Exception as e:
            raise Exception(f"Error calling LLM: {e}")
    
    async def acall_llm(self, system_prompt: str, user_prompt: str, response_format: Any = None) -> str:
        """
        Async version of call_llm
        
        Args:
            system_prompt: The system prompt to set context
            user_prompt: The user prompt/question
            response_format: Optional structured output format (Pydantic model)
            
        Returns:
            LLM response as string or structured object
        """
        try:
            # Create LLM with structured output if format is specified
            llm = self.llm
            if response_format is not None:
                llm = llm.with_structured_output(response_format)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("user", "{user_prompt}")
            ])
            
            # Create chain
            chain = prompt | llm

            self.invoke_model_info('async')
            
            # Async invoke chain
            response = await chain.ainvoke({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            })
            
            return response
            
        except Exception as e:
            raise Exception(f"Error calling LLM (async): {e}")
    
    def invoke_model_info(self, mode : Literal['async', 'sync']):
        provider_info = self.get_provider_info()
        model_name = provider_info.get('model')
        model_type = provider_info.get('type')

        print(f"Invoking the model {model_name} ({model_type}) in {mode} mode...")

        return

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider configuration"""
        info = {
            "type": self.llm_config.get('type'),
            "model": self.llm_config.get('model'),
            "temperature": self.llm_config.get('temperature', 0),
            "base_url": self.llm_config.get('base_url'),
            "has_api_key": bool(self.llm_config.get('api_key'))
        }
        
        # Add API version for Azure OpenAI
        if self.llm_config.get('type', '').lower() == 'azureopenai':
            info["api_version"] = self.llm_config.get('api_version', '2025-01-01-preview')
            
        return info
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working"""
        try:
            response = self.call_llm(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello' if you can read this message."
            )
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# Convenience function for quick LLM calls
def quick_llm_call(
    system_prompt: str, 
    user_prompt: str, 
    response_format: Any = None, 
    config_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> str:
    """
    Quick function to make an LLM call without initializing the class
    
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt
        response_format: Optional structured output format
        config_path: Path to configuration file (optional if config is provided)
        config: Configuration dictionary (optional if config_path is provided)
        
    Returns:
        LLM response
        
    Note:
        Either config_path or config must be provided.
        If neither is provided, defaults to "../config/config.yaml"
    """
    if config_path is None and config is None:
        config_path = "../config/config.yaml"
    
    provider = LLMProvider(config_path=config_path, config=config)
    return provider.call_llm(system_prompt, user_prompt, response_format)


