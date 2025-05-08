from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_tokenizer_and_model(model_name: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model
