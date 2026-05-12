import os
import torch
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# Default configuration
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

        Args:
            config: Dictionary with model and sampling parameters.
            cuda_device: CUDA device index to use (sets CUDA_VISIBLE_DEVICES).
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
                hf_token=os.getenv("HF_TOKEN"),
            )
            print(f"✅ vLLM model loaded successfully: {self.config.get('model')}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def generate(self, messages: list[dict]) -> str:
        """
        Generate a single response for a conversation (list of messages).

        Args:
            messages: List of chat messages dicts with 'role' and 'content'.

        Returns:
            Generated text string.
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
        Generate one response for each conversation in the batch.

        Args:
            batch_messages: List of conversations; each conversation is a list of message dicts.

        Returns:
            List of generated text strings, one per conversation.
        """
        outputs = self.model.chat(
            batch_messages,
            sampling_params=self.sampling_params,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        return [output.outputs[0].text for output in outputs]

    def generate_n_samples(self, messages: list[dict], n: int) -> list[str]:
        """
        Generate n responses for a single conversation using vLLM's native n-sampling.

        This is more efficient than calling generate() n times because it performs
        a single forward pass with n outputs.

        Args:
            messages: List of chat message dicts with 'role' and 'content'.
            n: Number of responses to generate.

        Returns:
            List of n generated text strings.
        """
        sampling_params_n = SamplingParams(
            n=n,
            temperature=self.config.get("temperature", 1.0),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 20),
            min_p=self.config.get("min_p", 0.0),
            presence_penalty=self.config.get("presence_penalty", 1.5),
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
            max_tokens=self.config.get("max_tokens", 1024),
        )
        outputs = self.model.chat(
            messages,
            sampling_params=sampling_params_n,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        if outputs and len(outputs) > 0:
            return [out.text for out in outputs[0].outputs]
        return []

    def generate_batch_n_samples(
        self, batch_messages: list[list[dict]], n: int
    ) -> list[list[str]]:
        """
        Generate n responses for each conversation in the batch.

        Each prompt gets n independent samples in a single batched call,
        maximising GPU throughput.

        Args:
            batch_messages: List of conversations; each conversation is a list of message dicts.
            n: Number of responses to generate per conversation.

        Returns:
            List of lists: outer list has one entry per conversation,
            inner list has n generated text strings.
        """
        sampling_params_n = SamplingParams(
            n=n,
            temperature=self.config.get("temperature", 1.0),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 20),
            min_p=self.config.get("min_p", 0.0),
            presence_penalty=self.config.get("presence_penalty", 1.5),
            repetition_penalty=self.config.get("repetition_penalty", 1.0),
            max_tokens=self.config.get("max_tokens", 1024),
        )
        outputs = self.model.chat(
            batch_messages,
            sampling_params=sampling_params_n,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        return [[out.text for out in output.outputs] for output in outputs]
