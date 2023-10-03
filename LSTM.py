import torch
import torch.nn as nn
import torch.nn.functional as F


class CharPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_id):
        super(CharPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        bidirectional = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.num_labels = vocab_size
        if bidirectional:
            self.fc_out = nn.Linear(hidden_dim*2, self.num_labels)
        else:
            self.fc_out = nn.Linear(hidden_dim, self.num_labels)

    def forward(self, x):
        # Embedding layer
        x = self.embedding(x)

        # LSTM layer
        _, (hidden_state, c_n) = self.lstm(x)
        # Four fully connected layers with ReLU activation
        hidden_state = torch.cat((hidden_state[-1, :, :], hidden_state[-2, :, :]), -1)
        hidden_state = self.dropout(hidden_state)

        out = F.relu(self.fc_out(hidden_state))

        return out
