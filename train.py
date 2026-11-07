import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pickle
from collections import Counter
from encoderModel import SpamClassifier, InputEmbeddings, MultiHeadAttentionBlock, FeedForwardBlock, EncoderBlock, Encoder

# Hypoer parameters
vocab_size = 500
seq_len = 64
d_model = 128
h = 4
d_ff = 512
dropout = 0.1
num_layers = 2
batch_size = 16
epochs = 5


class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, seq_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = float(self.labels[idx])
        
        tokens = text.lower().split()
        token_ids = [self.word2idx.get(word, 1) for word in tokens]

        if len(token_ids) < self.seq_len:
            token_ids = token_ids + [0] * (self.seq_len - len(token_ids)) # 0 = Padding token
        else:
            token_ids = token_ids[:self.seq_len]
            
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor([label], dtype=torch.float)

# Accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc


input_embed = InputEmbeddings(seq_len, vocab_size, d_model, dropout)
encoder_blocks = nn.ModuleList([
    EncoderBlock(
        features=d_model,
        self_attention_block=MultiHeadAttentionBlock(d_model, h, dropout),
        feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout),
        dropout=dropout
    ) for _ in range(num_layers)
])
encoder = Encoder(encoder_blocks, features=d_model)
model = SpamClassifier(encoder, input_embed, d_model)

# training loop
def main():       
    dataset = load_dataset("codesignal/sms-spam-collection")
    full_data = dataset["train"]
    split_dataset = full_data.train_test_split(test_size=0.2, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    def extract_texts_and_labels(split):
        texts = split["text"] if "text" in split.column_names else split["message"]
        raw_labels = split["label"] if "label" in split.column_names else split["category"]
        if isinstance(raw_labels[0], str):
            labels = [1.0 if l.lower() == 'spam' else 0.0 for l in raw_labels]
        else:
            labels = [float(l) for l in raw_labels]
        return texts, labels
    train_texts, train_labels = extract_texts_and_labels(train_data)
    val_texts, val_labels = extract_texts_and_labels(val_data)
    word_counts = Counter()
    
    for text in train_texts:
        word_counts.update(text.lower().split())
    most_common = word_counts.most_common(vocab_size - 2)
    word2idx = {"[PAD]": 0, "[UNK]": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx
    train_dataset = SimpleTextDataset(train_texts, train_labels, word2idx, seq_len)
    val_dataset = SimpleTextDataset(val_texts, val_labels, word2idx, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # train....
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_inputs, batch_labels in train_loader:
            mask = (batch_inputs != 0).unsqueeze(1).unsqueeze(2).float()

            optimizer.zero_grad()
            outputs = model(batch_inputs, mask)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            train_correct += torch.eq(batch_labels, preds).sum().item()
            train_total += batch_labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100

        model.eval() # Turn off dropout during evaluation
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # Save memory by not tracking gradients
            for batch_inputs, batch_labels in val_loader:
                mask = (batch_inputs != 0).unsqueeze(1).unsqueeze(2).float()
                
                outputs = model(batch_inputs, mask)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))
                val_correct += torch.eq(batch_labels, preds).sum().item()
                val_total += batch_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "spam_transformer.pth")
    
    with open("word2idx.pkl", "wb") as f:
        pickle.dump(word2idx, f)    
    print("Model and vocabulary saved successfully!")

if __name__ == "__main__":
    main()



# predicting something..

def predict_email(email_text, model, word2idx, seq_len=64, device="cpu"):
    model.eval()
    
    tokens = email_text.lower().split()
    token_ids = [word2idx.get(word, 1) for word in tokens] # 1 = [UNK] token
    
    if len(token_ids) < seq_len:
        token_ids = token_ids + [0] * (seq_len - len(token_ids))
    else:
        token_ids = token_ids[:seq_len]
        
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2).float()
    
    with torch.no_grad():
        logit = model(input_tensor, mask)
        probability = torch.sigmoid(logit).item()
        
    is_spam = probability >= 0.5
    label = "SPAM" if is_spam else "NOT SPAM (HAM)"
    confidence = probability if is_spam else (1.0 - probability)
    
    return label, confidence * 100

if __name__ == "__main__":
    sample_emails = [
        "Hey! Are we still meeting for lunch at 12:30 today? Let me know.",
        "URGENT: Your account has been suspended! Click here immediately to claim your $1000 prize bonus!!",
        "Hi John, attached is the project presentation update for tomorrow's team sync.",
        "Win a free iPhone 16! Congratulations user, you have been selected. Call 1-800-SPAM now!"
    ] 
    
    print("\n--- Testing Model on Custom Emails ---")
    for email in sample_emails:
        result, confidence = predict_email(email, model, word2idx)
        print(f"\nEmail: \"{email}\"")
        print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
