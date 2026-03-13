import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "./extraction_model/extraction_model"
VOCAB_FILE = MODEL_PATH + "/vocab.txt"

# Explicitly specify the vocab_file to avoid NoneType error
tokenizer = BertTokenizer.from_pretrained(
    MODEL_PATH,
    vocab_file=VOCAB_FILE,
    local_files_only=True
)


model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)
model.eval()


def predict_clause(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Get prediction and confidence
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_idx].item()

    # Get label name
    if hasattr(model.config, "id2label") and model.config.id2label:
        label = model.config.id2label[pred_idx]
    else:
        label = str(pred_idx)

    return label, confidence



if __name__ == "__main__":
    sample_text = "This Agreement shall be governed by the laws of England."
    clause_name, confidence = predict_clause(sample_text)

    print("Predicted clause:", clause_name)
    print(f"Confidence: {confidence:.4f}")