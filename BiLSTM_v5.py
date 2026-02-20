"""
BiLSTM v5: Bidirectional LSTM with Attention and Oversampling
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def preprocess_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = ' '.join(text.split())
    return text


class Vocabulary:
    """Build vocabulary from texts"""
    
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
    def build_vocab(self, texts, min_freq=2):
        """Build vocabulary from list of texts"""
        word_counts = Counter()
        for text in texts:
            word_counts.update(str(text).split())
        
        idx = 2
        for word, count in word_counts.items():
            if count >= min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        return self
        
    def encode(self, text):
        """Convert text to sequence of indices"""
        words = str(text).split()
        return [self.word_to_idx.get(word, 1) for word in words]
    
    def encode_batch(self, texts):
        """Encode multiple texts"""
        return [self.encode(text) for text in texts]


def pad_sequences(sequences, max_len):
    """Pad sequences to max_len"""
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important words"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_dim)
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_dim)
        return attended_output, attention_weights


class BiLSTMWithAttention(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, 
                 num_layers=2, num_classes=3, dropout=0.3):
        super(BiLSTMWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = AttentionLayer(hidden_dim * 2)
        
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.dropout1(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        attended_output, attention_weights = self.attention(lstm_out)
        
        out = self.dropout2(attended_output)
        out = self.relu(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc2(out)
        return out


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, accuracy, f1, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, labels, filename='confusion_matrix.png', title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def main():
    # ==================== CONFIGURATION ====================
    CHATGPT_CSV = 'final_data/labeled_political_sentiment.csv'
    HANDLABELED_CSV = 'final_data/hand_labeled_data.csv'
    TEXT_COLUMN = 'text'
    LABEL_COLUMN = 'sentiment_label'
    
    MAX_LEN = 200
    MIN_FREQ = 2
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30
    PATIENCE = 5
    CLIP_GRAD = 1.0
    # ======================================================
    print("LOADING DATA")
    
    df_chatgpt = pd.read_csv(CHATGPT_CSV)
    print(f"ChatGPT dataset shape: {df_chatgpt.shape}")
    
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df_chatgpt['label_encoded'] = df_chatgpt[LABEL_COLUMN].map(label_map)
    
    print(f"\nOriginal label distribution:")
    for label, count in df_chatgpt[LABEL_COLUMN].value_counts().items():
        print(f"  {label}: {count} ({count/len(df_chatgpt)*100:.1f}%)")
    
    # Stratified split
    train_df, temp_df = train_test_split(
        df_chatgpt, test_size=0.2, random_state=42, 
        stratify=df_chatgpt['label_encoded']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42,
        stratify=temp_df['label_encoded']
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    # OVERSAMPLING: Balance minority classes
    print("OVERSAMPLING MINORITY CLASSES")
    
    class_counts = train_df['label_encoded'].value_counts().sort_index()
    print(f"\nBefore oversampling:")
    for i, count in enumerate(class_counts):
        label_name = ['Negative', 'Neutral', 'Positive'][i]
        print(f"  {label_name}: {count} samples")
    
    target_size = class_counts.max()
    
    augmented_dfs = [train_df]
    for class_label in [0, 2]:  # Negative and Positive
        class_df = train_df[train_df['label_encoded'] == class_label]
        needed = target_size - len(class_df)
        
        if needed > 0:
            oversampled = class_df.sample(n=needed, replace=True, random_state=42)
            augmented_dfs.append(oversampled)
            label_name = 'Negative' if class_label == 0 else 'Positive'
            print(f"\nOversampled {label_name}: +{needed} samples")
    
    train_df_augmented = pd.concat(augmented_dfs, ignore_index=True)
    
    print(f"\nAfter oversampling:")
    for i in [0, 1, 2]:
        count = (train_df_augmented['label_encoded'] == i).sum()
        label_name = ['Negative', 'Neutral', 'Positive'][i]
        print(f"  {label_name}: {count} samples")
    
    print(f"\nTotal training samples: {len(train_df)} → {len(train_df_augmented)}")
    print(f"Increase: {(len(train_df_augmented)/len(train_df) - 1)*100:.1f}%")
    
    # Load hand-labeled
    df_handlabeled = pd.read_csv(HANDLABELED_CSV)
    df_handlabeled['label_encoded'] = df_handlabeled[LABEL_COLUMN].map(label_map)
    print(f"\nHand-labeled samples: {len(df_handlabeled)}")
    
    # Preprocessing
    print("PREPROCESSING")
    
    print("\nCleaning text data...")
    train_df_augmented[TEXT_COLUMN] = train_df_augmented[TEXT_COLUMN].apply(preprocess_text)
    val_df[TEXT_COLUMN] = val_df[TEXT_COLUMN].apply(preprocess_text)
    test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].apply(preprocess_text)
    df_handlabeled[TEXT_COLUMN] = df_handlabeled[TEXT_COLUMN].apply(preprocess_text)
    
    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_vocab(train_df_augmented[TEXT_COLUMN].values, min_freq=MIN_FREQ)
    
    print("Encoding sequences...")
    train_seqs = vocab.encode_batch(train_df_augmented[TEXT_COLUMN].values)
    val_seqs = vocab.encode_batch(val_df[TEXT_COLUMN].values)
    test_seqs = vocab.encode_batch(test_df[TEXT_COLUMN].values)
    handlabeled_seqs = vocab.encode_batch(df_handlabeled[TEXT_COLUMN].values)
    
    print(f"Padding sequences to max length {MAX_LEN}...")
    train_padded = pad_sequences(train_seqs, MAX_LEN)
    val_padded = pad_sequences(val_seqs, MAX_LEN)
    test_padded = pad_sequences(test_seqs, MAX_LEN)
    handlabeled_padded = pad_sequences(handlabeled_seqs, MAX_LEN)
    
    # Create datasets
    train_dataset = SentimentDataset(train_padded, train_df_augmented['label_encoded'].values)
    val_dataset = SentimentDataset(val_padded, val_df['label_encoded'].values)
    test_dataset = SentimentDataset(test_padded, test_df['label_encoded'].values)
    handlabeled_dataset = SentimentDataset(handlabeled_padded, df_handlabeled['label_encoded'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    handlabeled_loader = DataLoader(handlabeled_dataset, batch_size=BATCH_SIZE)
    
    # Model
    print("MODEL: BiLSTM")
    
    model = BiLSTMWithAttention(
        vocab_size=len(vocab.word_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"  Vocabulary: {len(vocab.word_to_idx):,}")
    print(f"  Embedding:  {EMBEDDING_DIM}-dim")
    print(f"  Hidden:     {HIDDEN_DIM}-dim BiLSTM")
    print(f"  Attention:  Learns word importance")
    print(f"  Total parameters: {num_params:,}")
    
    # Use lighter class weights since data is balanced
    # But still give slight boost to minority classes
    class_weights = torch.FloatTensor([
        1.5,  # Negative (slight boost)
        0.8,  # Neutral (slight reduction)
        1.2   # Positive (slight boost)
    ])
    print(f"\nClass weights (moderate): {class_weights.numpy()}")
    print("  (Using lighter weights since data is now balanced)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training
    print("TRAINING: ATTENTION + BALANCED DATA")
    
    best_val_f1 = 0
    best_val_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, CLIP_GRAD
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}", end='')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'bilstm_v5.pth')
            print(" ✓ BEST")
        else:
            epochs_no_improve += 1
            print()
        
        scheduler.step(val_f1)
        
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Evaluation
    print("FINAL EVALUATION")
    
    model.load_state_dict(torch.load('bilstm_v5.pth'))
    
    # ChatGPT Test Set
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print("CHATGPT TEST SET RESULTS")
    print(f"\nAccuracy:  {test_acc:.4f}")
    print(f"Macro F1:  {test_f1:.4f}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['Negative', 'Neutral', 'Positive'])}")
    
    plot_confusion_matrix(test_labels, test_preds,
                         ['Negative', 'Neutral', 'Positive'],
                         filename='confusion_matrix_v5_chatgpt_test.png',
                         title='v5: ChatGPT Test Set (Attention + Oversampling)')
    
    # Hand-labeled Test Set
    hand_loss, hand_acc, hand_f1, hand_preds, hand_labels = evaluate(
        model, handlabeled_loader, criterion, device
    )
    
    print("HAND-LABELED TEST SET RESULTS")
    print(f"\nAccuracy:  {hand_acc:.4f}")
    print(f"Macro F1:  {hand_f1:.4f}")
    print(f"\n{classification_report(hand_labels, hand_preds, target_names=['Negative', 'Neutral', 'Positive'])}")
    
    plot_confusion_matrix(hand_labels, hand_preds,
                         ['Negative', 'Neutral', 'Positive'],
                         filename='confusion_matrix_v5_handlabeled.png',
                         title='v5: Hand-Labeled (Attention + Oversampling)')
    
    # Summary
    print("PERFORMANCE SUMMARY")
    
    print(f"\nChatGPT Test Set:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Macro F1:  {test_f1:.4f}")
    
    print(f"\nHand-Labeled Set:")
    print(f"  Accuracy:  {hand_acc:.4f}")
    print(f"  Macro F1:  {hand_f1:.4f}")
    
    print(f"\nModel: bilstm_v5.pth")
    
    print("\n✓ Training complete!")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
