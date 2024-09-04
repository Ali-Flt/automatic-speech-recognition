from datasets import load_dataset
from transformers import WhisperProcessor
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load the Persian subset of the Common Voice dataset
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fa")

# Load the Whisper processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Define a function to preprocess the audio and text
def preprocess(batch):
    audio = torch.tensor(batch["audio"]["array"], dtype=torch.float)
    # Resample the audio to 16kHz if needed
    audio = torchaudio.transforms.Resample(orig_freq=batch["audio"]["sampling_rate"], new_freq=16000)(audio)
    
    # Prepare the input features for Whisper
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]

    # Tokenize the text
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Apply the preprocessing to the dataset
dataset = dataset.map(preprocess, remove_columns=["audio", "sentence", "client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"])



# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-fa-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    warmup_steps=500,
    save_total_limit=3,
    num_train_epochs=20,
    fp16=True,
    predict_with_generate=True,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    # report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
)

# Initialize the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
)

# Fine-tune the model
trainer.train()

metrics = trainer.evaluate(dataset["test"])
print(f"Test set metrics: {metrics}")

model.save_pretrained("./whisper-large-fa-finetuned")
processor.save_pretrained("./whisper-large-fa-finetuned")
model.push_to_hub("Aflt98/whisper-large-v3-fa-finetuned")
processor.push_to_hub("Aflt98/whisper-large-v3-fa-finetuned")
