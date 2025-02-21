from transformers import AutoTokenizer, AutoModelForCausalLM,Trainer, TrainingArguments
import Datasets



# 数据预处理
def preprocess_function(examples):
    # 这里将问题和答案转化为模型能够理解的格式
    inputs = tokenizer(examples['question'], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(examples['answer'], truncation=True, padding="max_length", max_length=128)
    inputs['labels'] = labels['input_ids']
    return inputs



model_path = input("Enter path to saved model: ")
dataset_url = input("Enter path to dataset: ")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

ds = Datasets.load_dataset(dataset_url)

# 处理数据集
train_dataset = ds['train'].map(preprocess_function, batched=True)
test_dataset = ds['test'].map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model(output_dir=r"E:\Ollma\Ollama\Models\manifests\registry.ollama.ai\library")