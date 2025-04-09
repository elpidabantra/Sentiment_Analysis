import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

from src import config, data_preprocessing, utils
from src.models.feedforward_nn import train_feedforward_model
from src.models.lstm_model import train_lstm_model

def main():
    # Check if CSV file exists; if so, load real data, otherwise use dummy data
    if os.path.exists(config.CSV_DATA_PATH):
        print(f"Loading data from {config.CSV_DATA_PATH} ...")
        X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = data_preprocessing.prepare_text_dataset(config.CSV_DATA_PATH)
    else:
        print("CSV file not found. Using dummy text data ...")
        from src.data_preprocessing import generate_dummy_text_data, generate_dummy_tfidf_data
        X_train_text = generate_dummy_text_data(config.NUM_SAMPLES)[:int(0.6*config.NUM_SAMPLES)]
        X_val_text = generate_dummy_text_data(config.NUM_SAMPLES)[int(0.6*config.NUM_SAMPLES):int(0.8*config.NUM_SAMPLES)]
        X_test_text = generate_dummy_text_data(config.NUM_SAMPLES)[int(0.8*config.NUM_SAMPLES):]
        # For dummy labels:
        y_train = np.random.randint(0, config.NUM_CLASSES, int(0.6 * config.NUM_SAMPLES))
        y_val = np.random.randint(0, config.NUM_CLASSES, int(0.2 * config.NUM_SAMPLES))
        y_test = np.random.randint(0, config.NUM_CLASSES, int(0.2 * config.NUM_SAMPLES))
    
    # For Feedforward NN (TF-IDF approach), you might want to generate TF-IDF features.
    # For this example, assume that you have a separate processing for numeric features.
    # Here, we will process text data for the LSTM model:
    
    # Tokenize text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train_text)
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_val_seq = tokenizer.texts_to_sequences(X_val_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    max_length = 20  # or adjust based on the actual data
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    # Build a Word2Vec model on the training data
    from gensim.models import Word2Vec
    X_train_tokenized = [text.lower().split() for text in X_train_text]
    word2vec_model = Word2Vec(sentences=X_train_tokenized, vector_size=300, window=5, min_count=1, workers=4)
    
    # Build the embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embed_dim = 300
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    
    # Convert padded sequences to torch tensors for LSTM model
    X_train_lstm = torch.tensor(X_train_padded, dtype=torch.long)
    X_val_lstm = torch.tensor(X_val_padded, dtype=torch.long)
    X_test_lstm = torch.tensor(X_test_padded, dtype=torch.long)
    
    # Convert labels to tensors (make sure your labels are numerical)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
    y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.long)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.long)
    
    # Create DataLoaders for LSTM model
    from torch.utils.data import DataLoader, TensorDataset
    batch_size_lstm = config.BATCH_SIZE_LSTM
    train_dataset_lstm = TensorDataset(X_train_lstm, y_train_tensor)
    val_dataset_lstm = TensorDataset(X_val_lstm, y_val_tensor)
    test_dataset_lstm = TensorDataset(X_test_lstm, y_test_tensor)
    train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=batch_size_lstm, shuffle=True)
    val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=batch_size_lstm, shuffle=False)
    test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=batch_size_lstm, shuffle=False)
    
    print("\nTraining LSTM Model (Word2Vec Embeddings)...")
    lstm_model, lstm_train_losses, lstm_val_losses, lstm_train_acc, lstm_val_acc = train_lstm_model(
        train_loader_lstm, val_loader_lstm, vocab_size, embed_dim, hidden_dim=128, num_classes=config.NUM_CLASSES,
        embedding_matrix=embedding_matrix, num_epochs=config.EPOCHS_LSTM, lr=config.LEARNING_RATE_LSTM
    )
    
    utils.plot_metrics(config.EPOCHS_LSTM, lstm_train_losses, lstm_val_losses, lstm_train_acc, lstm_val_acc, title_prefix="LSTM")
    
    lstm_model.eval()
    lstm_preds, lstm_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader_lstm:
            outputs = lstm_model(inputs)
            _, predicted = torch.max(outputs, 1)
            lstm_preds.extend(predicted.cpu().numpy())
            lstm_true.extend(labels.cpu().numpy())
    print("\nLSTM Model Classification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(lstm_true, lstm_preds))
    
if __name__ == "__main__":
    main()
