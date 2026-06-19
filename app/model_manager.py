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

# At/below this temperature, switch to greedy decoding (do_sample=False) so the
# sampler doesn't error on temperature ~= 0.
GREEDY_THRESHOLD = 0.05

# Confirmed model context facts. Carried in the context_building event payload
# (not yet displayed; bucket 06's gauge will consume them).
MAX_CONTEXT = 262144      # 256k via YaRN
NATIVE_CONTEXT = 16384    # 16k native train length; quality best within this


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
        # Only pass `tools=` when there are tools to advertise. An empty/none
        # tool set (e.g. all tools toggled off, or the forced final answer) must
        # NOT pass the dict-return kwarg — it isn't supported by
        # MistralCommonBackend.apply_chat_template. A bare tensor result is
        # wrapped into a dict below, which is all generate() needs.
        kwargs = {"return_tensors": "pt"}
        if tools:
            kwargs["tools"] = tools

        tokenized = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(tokenized, torch.Tensor):
            tokenized = {"input_ids": tokenized}
        tokenized["input_ids"] = tokenized["input_ids"].to("cuda")
        if "attention_mask" in tokenized:
            tokenized["attention_mask"] = tokenized["attention_mask"].to("cuda")
        return tokenized

    def decode_prompt(self, input_ids):
        """Decode the exact input_ids sent to the model back into text, for display.
        Special tokens are preserved so markers like [INST] / [AVAILABLE_TOOLS]
        appear as the model sees them."""
        ids = input_ids[0].tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def generate_streaming(self, tokenized, on_token=None, temperature=0.7):
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
            "streamer": streamer,
        }

        if temperature is None or temperature < GREEDY_THRESHOLD:
            # Greedy decoding; omit temperature so the sampler doesn't error.
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

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
