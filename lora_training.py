from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast, set_seed, BertForSequenceClassification
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

set_seed(42)

def train_lora(weight_path,save_path,lr):
    tokenizer = NllbTokenizerFast.from_pretrained(
        "facebook/nllb-200-distilled-1.3B", src_lang="kor_Hang", tgt_lang="eng_Latn"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B",device_map='auto')
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    from datasets import Dataset
    import json
    with open('./kor-eng_dataset.json','r')as f:
        a = json.load(f)

    dataset = Dataset.from_dict(a)
    dataset = dataset.train_test_split(test_size=0.015, shuffle=True)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples['korean'], text_target=examples['english'], max_length=90, truncation=True)
        return model_inputs

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    import numpy as np
    import evaluate
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    training_args = Seq2SeqTrainingArguments(
        output_dir=weight_path,
        evaluation_strategy="steps",
        save_strategy='steps',
        logging_steps=5000,
        eval_steps=25000,
        save_steps=25000,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=24,
        weight_decay=0.01,
        num_train_epochs=3,
        predict_with_generate=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_bleu',
        ddp_find_unused_parameters=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.save_pretrained(save_path)

if __name__ == '__main__' :
    train_lora('lora_training_nllb_1p3B_lr=1e3', 'lora_training_nllb_reverse_1p3B_lr=1e3_saved', lr=1e-3)
    # train_lora('lora_training_nllb_600m_lr=2.5e4','lora_training_nllb_600m_lr=2.5e4_saved',lr=2.5e-4)