"""
QLoRA fine-tuning of Mistral 7B Instruct for a structured extraction task
(e.g. MaintIE entities or failure modes from vehicle complaints).

Setup (pin versions; the TRL API changes often, so check its docs if an argument errors):
    pip install -U torch transformers datasets peft trl bitsandbytes accelerate

Before running:
    1. Accept the model's terms on its Hugging Face page, then: huggingface-cli login
    2. Prepare train.jsonl and val.jsonl (one example per line):
       {"messages": [
          {"role": "user", "content": "Extract entities as JSON from: brg repl o/s pump P12 noisy"},
          {"role": "assistant", "content": "{\"PhysicalObject\": [\"brg\", \"pump P12\"], ...}"}]}
       Note: Mistral's chat template has no system role, so put instructions in the user turn.
    3. Keep a separate, hand-checked test set that is never used for training.

Hardware: a 24 GB GPU (L4, A10G, e.g. SageMaker ml.g5.2xlarge) is comfortable.
A 16 GB T4 (free Colab) works with small batches but is slow.
"""

import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "mistral-7b-extractor-lora"

# bfloat16 on modern GPUs (A100, L4, A10G); float16 on older ones like the T4
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base_model():
    """Load Mistral 7B in 4-bit (the 'Q' in QLoRA) so it fits on a single GPU."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=DTYPE
    )
    return model, tokenizer


def train():
    model, tokenizer = load_base_model()
    data = load_dataset(
        "json", data_files={"train": "train.jsonl", "validation": "val.jsonl"}
    )

    # LoRA: train small adapter matrices instead of all 7B weights
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,               # 1-3 is typical; watch validation loss
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,    # effective batch size 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=1024,                  # raise if your inputs are longer
        gradient_checkpointing=True,      # saves memory at some speed cost
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        bf16=(DTYPE == torch.bfloat16),
        fp16=(DTYPE == torch.float16),
        report_to="none",                 # or "mlflow" / "wandb" for tracking
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)       # saves only the small LoRA adapter
    tokenizer.save_pretrained(OUTPUT_DIR)


def extract(text, model, tokenizer, instruction="Extract entities as JSON from: "):
    """Run the fine-tuned model on one input and parse its JSON output."""
    messages = [{"role": "user", "content": instruction + text}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=512, do_sample=False)
    answer = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return {"_parse_error": answer}   # count these in your evaluation


def load_finetuned():
    """Base model + trained adapter, for inference and evaluation."""
    model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    train()
