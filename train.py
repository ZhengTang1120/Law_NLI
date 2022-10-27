from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments
from trainer import *

from datasets import load_dataset, ClassLabel, load_metric

import torch
import numpy as np

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomModel(config, 3)

## loading dataset 
data_files = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
dataset = load_dataset("Tennessee/main/original", data_files=data_files)


classification_label = ClassLabel(num_classes = 3, names = ['refute','support','unrelated'])
dataset = dataset.cast_column('label', classification_label)

## add special tokens to the tokenizer and tokenize dataset
def tokenize(examples):
     return tokenizer(examples["text"], padding=True, truncation=True)

special_tok = ["[STATE]", "[LAW]", "[COND]", "[Person-1]", "[Person-2]", "[Person-3]", "[Person-4]", "[Person-5]", "[Person-6]", "[Person-7]", "[Person-8]", "[Person-9]", "[Person-10]", "[Person-11]", "[Person-12]", "[Person-13]", "[Person-14]", "[Person-15]", "[Person-16]", "[Person-17]", "[Person-18]", "[Person-19]", "[Person-20]", "[Address-1]", "[Address-2]", "[Organization-1]", "[Organization-2]", "[Organization-3]", "[Location-1]", "[Last name]", "[Number-1]", "[NO COND]"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tok})
tokenized_datasets = dataset.map(tokenize, batched = True)
model.encoder.resize_token_embeddings(len(tokenizer))
# reformat tokenized dataset (i.e., remove/rename certain columns)
tokenized_datasets = tokenized_datasets.remove_columns(['ID', 'text'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets = tokenized_datasets.with_format("torch")
## adding data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

## loading metrics and create compute_metric function
metric1 = load_metric("precision")
metric2 = load_metric("recall")
metric3 = load_metric("f1")
metric4 = load_metric("accuracy")

def compute_metrics(eval_pred):
     logits, labels = eval_pred
     print (len(logits), labels.shape)
     predictions = np.argmax(logits, axis=-1)
     precision = metric1.compute(predictions=predictions, references=labels, average="macro")
     recall = metric2.compute(predictions=predictions, references=labels, average="macro")
     f1 = metric3.compute(predictions=predictions, references=labels, average="macro")
     accuracy = metric4.compute(predictions=predictions, references=labels)
     return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# creating training_args and trainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.05,
    save_steps=630
)

trainer = CustomTrainer(
     model=model,
     args=training_args,
     train_dataset=tokenized_datasets['train'],
     eval_dataset=tokenized_datasets['dev'],
     tokenizer=tokenizer,
     data_collator=data_collator,
     compute_metrics = compute_metrics
 )

# training a model with the training args and trainer created above
trainer.train()
trainer.save_model('./results')
torch.cuda.empty_cache()
# evaluate the trained model with dev dataset
print(trainer.evaluate())