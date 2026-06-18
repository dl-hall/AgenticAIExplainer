import torch
import kagglehub

from transformers import (
        Mistral3ForConditionalGeneration,
        MistralCommonBackend
)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print(torch.cuda.get_device_name(0))

model_id = kagglehub.model_download(
    "mistral-ai/ministral-3/Transformers/ministral-3-3b-reasoning-2512"
)

tokenizer = MistralCommonBackend.from_pretrained(model_id)

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Quote a line from one of Shakespeare's plays.",
            }
        ],
    }
]

tokenized = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    return_dist=True,
)

tokenized["input_ids"] = tokenized["input_ids"].to("cuda")

if "attention_mask" in tokenized:
    tokenized["attention_mask"] = tokenized["attention_mask"].to("cuda")

with torch.inference_mode():
    output = model.generate(
        **tokenized,
        max_new_tokens=4096,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
    )[0]

print(
    tokenizer.decode(
        output[len(tokenized["input_ids"][0]) :],
        skip_special_tokens=True,
    )
)