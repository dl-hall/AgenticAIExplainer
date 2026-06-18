import torch
import kagglehub
import threading
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    TextIteratorStreamer,
)

MODELS = {
    "instruct": "mistral-ai/ministral-3/Transformers/ministral-3-3b-instruct-2512",
    "reasoning": "mistral-ai/ministral-3/Transformers/ministral-3-3b-reasoning-2512",
}


class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_key = None
        self._lock = threading.Lock()

    def load_model(self, model_key: str, on_progress=None):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from: {list(MODELS)}")

        with self._lock:
            if self.current_model_key == model_key:
                return

            if self.model is not None:
                if on_progress:
                    on_progress(f"Unloading {self.current_model_key} model...")
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()

            if on_progress:
                on_progress(f"Downloading/locating {model_key} model...")

            model_id = kagglehub.model_download(MODELS[model_key])

            if on_progress:
                on_progress(f"Loading tokenizer...")

            self.tokenizer = MistralCommonBackend.from_pretrained(model_id)

            if on_progress:
                on_progress(f"Loading model onto GPU (this may take a moment)...")

            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.current_model_key = model_key

            if on_progress:
                on_progress(f"Model '{model_key}' ready.")

    def tokenize(self, messages, tools=None):
        kwargs = {"return_tensors": "pt"}
        if tools:
            kwargs["tools"] = tools
        else:
            kwargs["return_dist"] = True

        tokenized = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(tokenized, torch.Tensor):
            tokenized = {"input_ids": tokenized}
        tokenized["input_ids"] = tokenized["input_ids"].to("cuda")
        if "attention_mask" in tokenized:
            tokenized["attention_mask"] = tokenized["attention_mask"].to("cuda")
        return tokenized

    def get_prompt_text(self, messages, tools=None):
        """Return the raw tokenized prompt as a string, for display in the UI."""
        kwargs = {"tokenize": False}
        if tools:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            return "(prompt text unavailable with tools)"

    def generate_streaming(self, tokenized, on_token=None):
        """Run generation in a background thread, yielding tokens via on_token callback.
        Returns the full generated text when complete."""

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        input_len = tokenized["input_ids"].shape[1]

        gen_kwargs = {
            **tokenized,
            "max_new_tokens": 4096,
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self._generate_thread,
            args=(gen_kwargs,),
            daemon=True,
        )
        thread.start()

        full_text = ""
        for token_text in streamer:
            full_text += token_text
            if on_token:
                on_token(token_text)

        thread.join()
        return full_text

    def _generate_thread(self, gen_kwargs):
        with torch.inference_mode():
            self.model.generate(**gen_kwargs)

    def decode_token_count(self, text):
        """Return the number of tokens in a text string."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
