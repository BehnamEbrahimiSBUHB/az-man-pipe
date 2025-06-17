from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import requests
import os
import logging

logger = logging.getLogger("azure_openai_pipeline")
logging.basicConfig(level=logging.INFO)

class Pipeline:
    class Valves(BaseModel):
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_VERSION: str
        AZURE_OPENAI_MODELS: str
        AZURE_OPENAI_MODEL_NAMES: str

    def __init__(self):
        """Initializes the Azure OpenAI pipeline."""
        self.type = "manifold"
        self.name = "Azure OpenAI: "
        self.valves = self.Valves(
            AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
            AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
            AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            AZURE_OPENAI_MODELS=os.getenv("AZURE_OPENAI_MODELS", "gpt-35-turbo;gpt-4o"),
            AZURE_OPENAI_MODEL_NAMES=os.getenv("AZURE_OPENAI_MODEL_NAMES", "GPT-35 Turbo;GPT-4o"),
        )
        self.set_pipelines()

    def set_pipelines(self):
        """Sets up available pipelines based on environment variables."""
        models = self.valves.AZURE_OPENAI_MODELS.split(";")
        model_names = self.valves.AZURE_OPENAI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model.strip(), "name": name.strip()} for model, name in zip(models, model_names)
        ]
        logger.info(f"azure_openai_manifold_pipeline - models: {self.pipelines}")

    async def on_valves_updated(self):
        self.set_pipelines()

    async def on_startup(self):
        logger.info(f"on_startup:{__name__}")

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{__name__}")

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator, dict]:
        """
        Send a chat completion request to Azure OpenAI.

        Args:
            user_message (str): The user's message.
            model_id (str): The deployed model ID.
            messages (List[dict]): The chat history.
            body (dict): The full OpenAI API request body.

        Returns:
            Union[str, Generator, Iterator, dict]: The response or an error message.
        """
        logger.info(f"pipe:{__name__}")
        logger.debug(f"Messages: {messages}")
        logger.debug(f"User message: {user_message}")

        headers = {
            "api-key": self.valves.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        url = f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{model_id}/chat/completions?api-version={self.valves.AZURE_OPENAI_API_VERSION}"

        allowed_params = {
            'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
            'enhancements', 'dataSources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
            'frequency_penalty', 'logit_bias', 'user', 'function_call', 'functions', 'tools',
            'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'
        }
        r = None
        # Remap user field if needed
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"].get("id", str(body["user"]))
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}
        dropped = set(body.keys()) - set(filtered_body.keys())
        if dropped:
            logger.warning(f"Dropped params: {', '.join(dropped)}")

        try:
            r = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=filtered_body.get("stream", False),
                timeout=30,  # Add a timeout
            )
            r.raise_for_status()
            if filtered_body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            error_text = getattr(r, "text", "")
            logger.error(f"Error in pipe: {e} ({error_text})")
            return f"Error: {e} ({error_text})"