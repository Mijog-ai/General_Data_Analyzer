# utils/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LSTMPumpModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        #super(LSTMPumpModel, self).__init__()
        #self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        #self.fc = nn.Linear(hidden_dim, output_dim)

        super(LSTMPumpModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 3
        self.bidirectional = False

        # Define a 1D Convolutional Layer before LSTM
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # Define LSTM layer with multiple stacked layers and optional bidirectionality
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, batch_first=True,
                            dropout=0.3 if 3 > 1 else 0, bidirectional=False)

        # Compute output feature size considering bidirectionality
        lstm_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # Additional fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Permute to match Conv1d input format (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Convert back for LSTM input

        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)

        # Pass through first FC layer
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)

        # Final output layer
        predictions = self.fc2(x)  # Shape: (batch_size, seq_len, output_dim)
        return predictions

def train_model(df, input_dim=2, hidden_dim=64, output_dim=2, epochs=100):
    model = LSTMPumpModel(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(df[[ "Speed", "Flow"]].values, dtype=torch.float32).unsqueeze(0)
    y_train = torch.tensor(df[["Pressure1", "Pressure2"]].values, dtype=torch.float32).unsqueeze(0)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train[:, -1])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = LSTMPumpModel(2, 64, 2)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model