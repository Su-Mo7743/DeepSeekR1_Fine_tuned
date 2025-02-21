# -*- coding: utf-8 -*-
"""DeepSeek1.5bSFT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19hFT01KnJ6pAzcOrxwXR81HW3g6ScyZt
"""

!pip install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

!pip install -q datasets trl transformers

from google.colab import userdata
from huggingface_hub import login
login(userdata.get('HF_API'))

from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = userdata.get('HF_API')
)

for name, module in model.named_modules():
    print(name)

#Config lora (Low-rank)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj'
        ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state = 3407,
    use_rslora = False,
    loftq_config = None

    )

#This model template using for the training
train_prompt_style = """Below is an instruction that describes a task, paired with an input, both english and Chinese version, that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step answer to ensure a logical and accurate response.
### Instruction:
You are a math expert with advanced knowledge in algebra.
Please answer the following math question.
### Question:
{}
{}

### Response:
<think>
{}
</think>
{}"""

#format the data
EOS_token = tokenizer.eos_token

def formatting_propmts_func(examples):
  input_en = examples['question']
  input_cn = examples['question_zh-cn']
  detailed_ans = examples['answer']
  dir_ans = examples['answer_only']
  text = []
  for input_en,input_cn,detailed_ans,dir_ans in zip(input_en,input_cn,detailed_ans,dir_ans):
    prompt = train_prompt_style.format(input_en,input_cn,detailed_ans,dir_ans)+ EOS_token
    text.append(prompt)
  return  {'text': text}

#Pull data from the HF
# im only taking 100 records
from datasets import load_dataset
data = load_dataset("swulling/gsm8k_chinese",split='train[:1000]', trust_remote_code=True)
dataset = data.map(formatting_propmts_func,batched = True)

# check the how the data looklike
dataset['text'][0]

# Do supervised-finetuning using the trl and config the parameter
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args=TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,# Use num_train_epochs = 1, warmup_ratio for full training runs!
        num_train_epochs = 3,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate=2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported (),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to = "none", # Use this for WandB etc
    )
  )

training_stats = trainer.train()

prompt_style = """Below is an instruction that describes a task, paired with an input, both english and Chinese version, that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step answer to ensure a logical and accurate response.

### Instruction:
You are a math expert with advanced knowledge in algebra.
Please answer the following math question.
### Question:
{}

### Response:
<think>
{}
</think>"""
#sample question
question = "莎莉在小学任教，并获得 320 美元用于为学生购买书籍。一本阅读书售价 12 美元，她的班上有 30 名学生。不幸的是，如果学校给她的书费不够，她就需要自掏腰包。莎莉需要自掏腰包多少钱才能给每个学生买一本阅读书？"

# Let's inference to see how our finetuned model looklike take up to max 1 minites
FastLanguageModel.for_inference(model)
input = tokenizer([prompt_style.format(question,"")],return_tensors='pt').to('cuda')
outputs = model.generate(
    input_ids = input.input_ids,
    attention_mask = input.attention_mask,
    max_new_tokens = 1200,
    eos_token_id = tokenizer.eos_token_id,
    use_cache = True
)
response = tokenizer.batch_decode(outputs,skip_special_tokens=True)
print(response[0].split("### Response:")[1])