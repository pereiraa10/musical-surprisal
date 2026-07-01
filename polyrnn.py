import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict

# --- Model Definition: PolyRNN ---

class PolyRNN(nn.Module):
    """
    Recurrent Neural Network designed to model polyphonic musical expectations.
    
    Based on the paper: "PolyRNN: A time-resolved model of polyphonic musical 
    expectations aligned with human brain responses" (Robert et al., 2024).
    
    The model processes piano-roll representations (88 piano keys) and outputs
    three key metrics:
    - Surprise (prediction error): difference between predicted and actual notes
    - Uncertainty (entropy): uncertainty among multiple note possibilities
    - Predicted density: likelihood of note events occurring
    
    Architecture:
    - Single LSTM layer with 88 hidden units
    - Input/output size of 88 (corresponding to piano keys A0-C8)
    - Dropout rate of 0.05
    - Sigmoid activation for probability outputs
    """
    
    def __init__(self, input_size=88, hidden_size=88, output_size=88, dropout_rate=0.05):
        super(PolyRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Single LSTM layer as specified in the paper
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Linear layer to project LSTM output to note probabilities
        self.linear = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation for probability outputs
        self.activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the PolyRNN model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 88)
               Piano-roll representation where each timestep is a binary vector
               indicating which notes are active
            hidden: Optional tuple of (h_0, c_0) hidden states for LSTM
                   If None, will be initialized to zeros
        
        Returns:
            predictions: Predicted probabilities for each note at each timestep
                        Shape: (batch_size, sequence_length, 88)
            hidden: Tuple of (h_n, c_n) final hidden states
        """
        # Pass through LSTM layer
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply linear transformation
        output = self.linear(lstm_out)
        
        # Apply sigmoid activation to get probabilities
        predictions = self.activation(output)
        
        return predictions, hidden
    
    def compute_musical_features(self, 
                                 predictions: torch.Tensor, 
                                 targets: torch.Tensor
                                 ) -> Dict[str, torch.Tensor]:
        """
        Compute the three key musical features from model predictions.
        
        Args:
            predictions: Model output probabilities, shape (batch, seq_len, 88)
            targets: Ground truth binary piano-roll, shape (batch, seq_len, 88)
        
        Returns:
            Dictionary containing:
            - 'surprise': Prediction error (cross-entropy per note)
            - 'uncertainty': Entropy of predictions
            - 'predicted_density': Expected number of active notes
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        predictions_clipped = torch.clamp(predictions, eps, 1 - eps)
        
        # Surprise: Binary cross-entropy (prediction error)
        # Computed per note, then summed across notes
        surprise = -(targets * torch.log2(predictions_clipped) + 
                    (1 - targets) * torch.log2(1 - predictions_clipped))
        surprise = surprise.sum(dim=-1)  # Sum across 88 notes
        
        # Uncertainty: Entropy of the predicted distribution
        # H(p) = -Σ[p*log(p) + (1-p)*log(1-p)]
        uncertainty = -(predictions_clipped * torch.log2(predictions_clipped) + 
                       (1 - predictions_clipped) * torch.log2(1 - predictions_clipped))
        uncertainty = uncertainty.sum(dim=-1)  # Sum across 88 notes
        
        # Predicted density: Expected number of notes (sum of probabilities)
        predicted_density = predictions.sum(dim=-1)
        
        return {
            'surprise': surprise,
            'uncertainty': uncertainty,
            'predicted_density': predicted_density
        }
    
    def init_hidden(self, batch_size: int, device: torch.device = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Batch size for hidden state
            device: Device to create tensors on
        
        Returns:
            Tuple of (h_0, c_0) initial hidden states
        """
        if device is None:
            device = next(self.parameters()).device
            
        h_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        
        return (h_0, c_0)


# --- Training Configuration ---

class PolyRNNTrainer:
    """
    Trainer class for PolyRNN model with Truncated Backpropagation Through Time (TBPTT).
    
    Training parameters based on the paper:
    - Batch size: 15
    - Batch length: 60 seconds (1200 timesteps at 20 Hz)
    - TBPTT window: 5 seconds (100 timesteps at 20 Hz)
    - Sampling frequency: 20 Hz (50 ms timestep)
    - Learning rate: 1e-5
    - Optimizer: Adam
    - Loss: Binary Cross-Entropy (log base 2)
    """
    
    def __init__(self, 
                 model: PolyRNN,
                 learning_rate: float = 1e-5,
                 batch_size: int = 15,
                 batch_length_sec: float = 60.0,
                 tbptt_window_sec: float = 5.0,
                 sampling_frequency: int = 20):
        
        self.model = model
        self.batch_size = batch_size
        
        # Convert time durations to number of timesteps
        self.batch_length = int(batch_length_sec * sampling_frequency)  # 1200 timesteps
        self.tbptt_window = int(tbptt_window_sec * sampling_frequency)  # 100 timesteps
        self.sampling_frequency = sampling_frequency
        
        # Loss function: Binary Cross-Entropy
        # Note: PyTorch BCELoss uses natural log, we'll convert to log2 in loss computation
        self.criterion = nn.BCELoss(reduction='mean')
        
        # Optimizer: Adam with learning rate 1e-5
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.device = next(model.parameters()).device
        
    def compute_bce_log2(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Cross-Entropy with log base 2 (as specified in paper).
        
        BCE_log2 = -1/N * Σ[y*log2(p) + (1-y)*log2(1-p)]
        """
        eps = 1e-10
        predictions_clipped = torch.clamp(predictions, eps, 1 - eps)
        
        # Compute BCE with log2
        bce_log2 = -(targets * torch.log2(predictions_clipped) + 
                    (1 - targets) * torch.log2(1 - predictions_clipped))
        
        return bce_log2.mean()
    
    def train_epoch(self, data_loader) -> Dict[str, float]:
        """
        Train for one epoch using Truncated Backpropagation Through Time (TBPTT).
        
        Args:
            data_loader: DataLoader providing batches of piano-roll sequences
                        Each batch should have shape (batch_size, sequence_length, 88)
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            # batch_data shape: (batch_size, sequence_length, 88)
            batch_data = batch_data.to(self.device)
            
            # Initialize hidden state for this batch
            hidden = self.model.init_hidden(batch_data.size(0), self.device)
            
            batch_loss = 0.0
            
            # Process sequence in TBPTT windows
            for i in range(0, batch_data.size(1) - 1, self.tbptt_window):
                # Get window of data
                end_idx = min(i + self.tbptt_window, batch_data.size(1) - 1)
                input_seq = batch_data[:, i:end_idx, :]
                target_seq = batch_data[:, i+1:end_idx+1, :]
                
                # Forward pass
                predictions, hidden = self.model(input_seq, hidden)
                
                # Detach hidden state for truncated BPTT
                hidden = tuple(h.detach() for h in hidden)
                
                # Compute loss
                loss = self.compute_bce_log2(predictions, target_seq)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (good practice for RNNs)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                batch_loss += loss.item()
            
            total_loss += batch_loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Evaluate model on validation/test data.
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)
                
                # Initialize hidden state
                hidden = self.model.init_hidden(batch_data.size(0), self.device)
                
                # Forward pass on entire sequence
                input_seq = batch_data[:, :-1, :]
                target_seq = batch_data[:, 1:, :]
                
                predictions, _ = self.model(input_seq, hidden)
                
                # Compute loss
                loss = self.compute_bce_log2(predictions, target_seq)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}


# --- Example Usage ---

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Instantiate the model
    model = PolyRNN(
        input_size=88,
        hidden_size=88,
        output_size=88,
        dropout_rate=0.05
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = PolyRNNTrainer(
        model=model,
        learning_rate=1e-5,
        batch_size=15,
        batch_length_sec=60.0,
        tbptt_window_sec=5.0,
        sampling_frequency=20
    )
    
    # Example: Create dummy data for demonstration
    # In practice, you would load MIDI data from the MAESTRO dataset
    batch_size = 15
    sequence_length = 1200  # 60 seconds at 20 Hz
    input_size = 88
    
    # Create dummy piano-roll data (random binary vectors)
    dummy_data = torch.bernoulli(torch.ones(batch_size, sequence_length, input_size) * 0.1)
    
    print(f"\nExample forward pass:")
    print(f"Input shape: {dummy_data.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(batch_size, device)
        dummy_data = dummy_data.to(device)
        
        predictions, hidden = model(dummy_data, hidden)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Hidden state shapes: h={hidden[0].shape}, c={hidden[1].shape}")
        
        # Compute musical features
        features = model.compute_musical_features(predictions, dummy_data)
        
        print(f"\nMusical Features (averaged over batch and time):")
        print(f"  Surprise (prediction error): {features['surprise'].mean().item():.4f} bits")
        print(f"  Uncertainty (entropy): {features['uncertainty'].mean().item():.4f} bits")
        print(f"  Predicted density: {features['predicted_density'].mean().item():.4f} notes")
    
    print("\n" + "="*70)
    print("PolyRNN Implementation Complete!")
    print("="*70)
    print("\nTo train the model on MAESTRO dataset:")
    print("1. Download MAESTRO dataset: https://magenta.tensorflow.org/datasets/maestro")
    print("2. Convert MIDI files to piano-roll format (binary arrays)")
    print("3. Create PyTorch DataLoader with the converted data")
    print("4. Call trainer.train_epoch(data_loader) for training")
    print("="*70)