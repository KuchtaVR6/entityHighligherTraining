import torch
from tqdm import tqdm

def compute_logits(model, infer_dataset, tokenizer):
    model.eval()

    for batch in tqdm(infer_dataset):
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])

        zero_tensor = torch.zeros_like(input_ids)

        tokens = [tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in input_ids]

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=zero_tensor)

            print(tokens, logits.shape)
