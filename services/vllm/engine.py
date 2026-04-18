import os
import torch
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# Default configuration copied from the notebook
DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "max_tokens": 1024,
    "enable_thinking": False,
}

class VLLMEngine:
    def __init__(self, config: dict = None, cuda_device: str = "1"):
        """
        Initialize the vLLM Engine.
        """
        load_dotenv()
        
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
            
        self.config = config or DEFAULT_CONFIG
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")

        self.enable_thinking = self.config.get("enable_thinking", False)
        self.sampling_params = SamplingParams(
            temperature=self.config.get("temperature", 1.0),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 20),
            min_p=self.config.get("min_p", 0.0),
            presence_penalty=self.config.get("presence_penalty", 1.5),
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
            max_tokens=self.config.get("max_tokens", 1024),
        )
        
        try:
            self.model = LLM(
                model=self.config.get("model"),
                tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
                gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.85),
                max_model_len=self.config.get("max_model_len", 4096),
                hf_token=os.getenv("HF_TOKEN")
            )
            print(f"✅ vLLM model loaded successfully: {self.config.get('model')}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def generate(self, messages: list[dict]) -> str:
        """
        Generate a response for a single conversation (list of messages).
        """
        outputs = self.model.chat(
            messages,
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text
        return ""

    def generate_batch(self, batch_messages: list[list[dict]]) -> list[str]:
        """
        Generate responses for a batch of conversations.
        """
        outputs = self.model.chat(
            batch_messages,
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        return [output.outputs[0].text for output in outputs]
