import torch
import torch.nn as nn
import torch.nn.functional as F

class StreamingSightMu(nn.Module):
    """
    Advanced gated recurrent model for skeleton-based gesture recognition 
    with joint attention mechanism
    """
    def __init__(self, input_dim, output_dim, hidden_size=512, dropout=0.3, num_layers=6):
        super(StreamingSightMu, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Joint attention mechanism - assigns different weights to different skeleton joints
        self.joint_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 22),  # Assuming 22 joints
            nn.Softmax(dim=-1)
        )
        
        # Gated recurrent units - similar to GRU gating mechanism
        # Update gates
        self.update_gates = nn.ModuleList([
            nn.Linear(hidden_size + hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Reset gates
        self.reset_gates = nn.ModuleList([
            nn.Linear(hidden_size + hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Candidate hidden state generation
        self.candidate_layers = nn.ModuleList([
            nn.Linear(hidden_size + hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout)
            for _ in range(num_layers)
        ])
        
        # Residual connections
        self.residual_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers - 1)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.output_activation = nn.ReLU()
        self.output_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        """
        x: Input tensor, shape [batch_size, seq_length=1, input_size]
        h: External hidden state, shape [num_layers, batch_size, hidden_size]
        """
        batch_size, seq_length, input_size = x.size()
        x = x.squeeze(1)  # [batch_size, input_size]
        
        # Apply joint attention - identify important joints
        joint_weights = self.joint_attention(x)  # [batch_size, 22]
        # Apply joint weights to the input features
        # Assuming input x is organized as [batch_size, 22*3], with each joint having 3 coordinates
        x_reshaped = x.view(batch_size, 22, 3)
        x_weighted = x_reshaped * joint_weights.unsqueeze(-1)
        x = x_weighted.reshape(batch_size, input_size)
        
        # Input projection
        x = self.input_projection(x)
        
        # Process through multiple gated layers
        new_h = []
        
        for i in range(self.num_layers):
            # Get current layer's hidden state
            curr_h = h[i]
            
            # Concatenate input and hidden state
            combined = torch.cat([x, curr_h], dim=-1)
            
            # Calculate update and reset gates
            update_gate = torch.sigmoid(self.update_gates[i](combined))
            reset_gate = torch.sigmoid(self.reset_gates[i](combined))
            
            # Calculate candidate hidden state
            reset_hidden = reset_gate * curr_h
            candidate_input = torch.cat([x, reset_hidden], dim=-1)
            candidate_hidden = torch.tanh(self.candidate_layers[i](candidate_input))
            
            # Update hidden state
            new_hidden = (1 - update_gate) * curr_h + update_gate * candidate_hidden
            
            # Apply dropout
            new_hidden = self.dropouts[i](new_hidden)
            
            # Save new hidden state
            new_h.append(new_hidden.detach())
            
            # Prepare input for next layer, adding residual connection
            if i < self.num_layers - 1:
                x = x + self.residual_projections[i](new_hidden)
            else:
                x = new_hidden
        
        # Final output layer
        x = self.output_projection(x)
        x = self.output_activation(x)
        x = self.output_dropout(x)
        output = self.fc(x)
        
        # Return output and updated hidden state
        return F.log_softmax(output, dim=-1), new_h

class StreamingSightBi(nn.Module):
    """
    Bidirectional LSTM model for skeleton-based gesture recognition
    """
    def __init__(self, input_dim, output_dim, hidden_size=256, dropout=0.2, num_layers=1):
        super(StreamingSightBi, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Output layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, hPt, cPt):
        # x: Input tensor, shape [batch_size, seq_length, input_size]
        # hPt, cPt: LSTM hidden state and cell state
        
        # Pass through LSTM
        x, (hPt, cPt) = self.lstm(x, (hPt, cPt))
        
        # Flatten and pass through fully connected layer
        x = self.flatten(x)
        x = F.relu(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1), (hPt, cPt)

class StreamingSightMuOriginal(nn.Module):
    """
    Original streaming sight model using fully connected layers with recurrent state
    """
    def __init__(self, input_dim, output_dim, hidden_size=1024, dropout=0.2, num_layers=3):
        super(StreamingSightMuOriginal, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Recurrent fully connected layers
        self.rnnsfc1 = nn.Linear(input_dim + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout)
        
        # Output layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        # x: Input tensor, shape [batch_size, seq_length, input_size]
        # h: External hidden state, shape [3, batch_size, hidden_size]
        
        batch_size, seq_length, input_size = x.size()
        x = x.squeeze(1)  # Assuming seq_length=1
        
        # First fully connected layer, input is concatenation of current input and hidden state
        xh1 = torch.cat([x, h[0]], dim=-1)
        nh0 = F.relu(self.rnnsfc1(xh1))
        nh0 = self.dropout1(nh0)
        
        # Second fully connected layer
        xh2 = torch.cat([nh0, h[1]], dim=-1)
        nh1 = F.relu(self.rnnsfc2(xh2))
        nh1 = self.dropout2(nh1)
        
        # Third fully connected layer
        xh3 = torch.cat([nh1, h[2]], dim=-1)
        nh2 = F.relu(self.rnnsfc3(xh3))
        nh2 = self.dropout3(nh2)
        
        # Flatten and pass through fully connected layer
        x = self.flatten(nh2)
        x = F.relu(x)
        x = self.fc(x)
        
        # Generate new hidden state
        hnew = [nh0.detach(), nh1.detach(), nh2.detach()]
        
        # Return output and updated hidden state
        return F.log_softmax(x, dim=-1), hnew

class TCRN(nn.Module):
    """
    Temporal Convolutional Recurrent Network for skeleton-based gesture recognition
    
    Combines temporal convolution with recurrent processing
    """
    def __init__(self, input_dim, output_dim, hidden_size=1024, dropout=0.2, num_layers=3, frames_per_segment=3):
        super(TCRN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frames_per_segment = frames_per_segment  # Number of frames to process at a time
        
        # Temporal convolution part - processes frames_per_segment consecutive frames
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=frames_per_segment, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # Recurrent fully connected layers - same structure as StreamingSightMuOriginal
        self.rnnsfc1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout)
        
        # Output layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        """
        x: Input tensor, shape [batch_size, frames_per_segment, input_size]
        h: External hidden state, shape [3, batch_size, hidden_size]
        """
        batch_size, seq_length, input_size = x.size()
        
        # Ensure correct number of input frames
        assert seq_length == self.frames_per_segment, f"Input should have {self.frames_per_segment} frames, but got {seq_length}"
        
        # Reshape tensor to fit Conv1d input requirements [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, input_size, frames_per_segment]
        
        # Apply temporal convolution
        # Output shape: [batch_size, hidden_size, 1]
        x = self.temporal_conv(x)
        
        # Squeeze last dimension
        x = x.squeeze(-1)  # [batch_size, hidden_size]
        
        # First fully connected layer, input is concatenation of convolution output and hidden state
        xh1 = torch.cat([x, h[0]], dim=-1)
        nh0 = F.relu(self.rnnsfc1(xh1))
        nh0 = self.dropout1(nh0)
        
        # Second fully connected layer
        xh2 = torch.cat([nh0, h[1]], dim=-1)
        nh1 = F.relu(self.rnnsfc2(xh2))
        nh1 = self.dropout2(nh1)
        
        # Third fully connected layer
        xh3 = torch.cat([nh1, h[2]], dim=-1)
        nh2 = F.relu(self.rnnsfc3(xh3))
        nh2 = self.dropout3(nh2)
        
        # Flatten and pass through fully connected layer
        x = self.flatten(nh2)
        x = F.relu(x)
        x = self.fc(x)
        
        # Generate new hidden state
        hnew = [nh0.detach(), nh1.detach(), nh2.detach()]
        
        # Return output and updated hidden state
        return F.log_softmax(x, dim=-1), hnew

class MSTCRN(nn.Module):
    """
    Multi-Scale Temporal Convolutional Recurrent Network
    
    Features:
    1. Introduces hierarchical external states: Fast and Slow layers
    2. Fast layer updates every frame, capturing short-term dependencies
    3. Slow layer updates every few frames, storing long-term information
    """
    def __init__(self, input_dim, output_dim, hidden_size=1024, dropout=0.2, 
                 num_layers=3, frames_per_segment=3, slow_update_rate=10):
        super(MSTCRN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frames_per_segment = frames_per_segment  # Number of frames to process at a time
        self.slow_update_rate = slow_update_rate     # Slow layer update frequency
        
        # Temporal convolution part - processes frames_per_segment consecutive frames
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=frames_per_segment, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # Fast layer recurrent fully connected - updates every step
        self.fast_rnnsfc1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.fast_dropout1 = nn.Dropout(p=dropout)
        self.fast_rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.fast_dropout2 = nn.Dropout(p=dropout)
        self.fast_rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.fast_dropout3 = nn.Dropout(p=dropout)
        
        # Slow layer update network - updates every slow_update_rate steps
        self.slow_update1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.slow_dropout1 = nn.Dropout(p=dropout)
        self.slow_update2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.slow_dropout2 = nn.Dropout(p=dropout)
        self.slow_update3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.slow_dropout3 = nn.Dropout(p=dropout)
        
        # Output layer - combines fast and slow states
        self.output_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, state_dict):
        """
        x: Input tensor, shape [batch_size, frames_per_segment, input_size]
        state_dict: Dictionary containing external states, structure:
            {
                'fast': [h1, h2, h3],  # Fast layer states
                'slow': [h1, h2, h3],  # Slow layer states
                'step_count': int       # Current step count, used to determine when to update slow layer
            }
        """
        batch_size, seq_length, input_size = x.size()
        
        # Ensure correct number of input frames
        assert seq_length == self.frames_per_segment, f"Input should have {self.frames_per_segment} frames, but got {seq_length}"
        
        # Reshape tensor to fit Conv1d input requirements [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, input_size, frames_per_segment]
        
        # Apply temporal convolution to get z_t
        z_t = self.temporal_conv(x)
        z_t = z_t.squeeze(-1)  # [batch_size, hidden_size]
        
        # Get current states
        fast_h = state_dict['fast']  # Fast layer hidden states
        slow_h = state_dict['slow']  # Slow layer hidden states
        step_count = state_dict['step_count']  # Current step count
        
        # Update fast layer - updates every step
        # First layer
        fast_xh1 = torch.cat([z_t, fast_h[0]], dim=-1)
        fast_nh0 = F.relu(self.fast_rnnsfc1(fast_xh1))
        fast_nh0 = self.fast_dropout1(fast_nh0)
        
        # Second layer
        fast_xh2 = torch.cat([fast_nh0, fast_h[1]], dim=-1)
        fast_nh1 = F.relu(self.fast_rnnsfc2(fast_xh2))
        fast_nh1 = self.fast_dropout2(fast_nh1)
        
        # Third layer
        fast_xh3 = torch.cat([fast_nh1, fast_h[2]], dim=-1)
        fast_nh2 = F.relu(self.fast_rnnsfc3(fast_xh3))
        fast_nh2 = self.fast_dropout3(fast_nh2)
        
        # Update slow layer - only when step_count is a multiple of slow_update_rate
        if step_count % self.slow_update_rate == 0:
            # First layer - slow layer uses fast layer state for updating
            slow_xh1 = torch.cat([fast_nh0, slow_h[0]], dim=-1)
            slow_nh0 = F.relu(self.slow_update1(slow_xh1))
            slow_nh0 = self.slow_dropout1(slow_nh0)
            
            # Second layer
            slow_xh2 = torch.cat([fast_nh1, slow_h[1]], dim=-1)
            slow_nh1 = F.relu(self.slow_update2(slow_xh2))
            slow_nh1 = self.slow_dropout2(slow_nh1)
            
            # Third layer
            slow_xh3 = torch.cat([fast_nh2, slow_h[2]], dim=-1)
            slow_nh2 = F.relu(self.slow_update3(slow_xh3))
            slow_nh2 = self.slow_dropout3(slow_nh2)
        else:
            # No update, maintain original state
            slow_nh0 = slow_h[0]
            slow_nh1 = slow_h[1]
            slow_nh2 = slow_h[2]
        
        # Combine final states of fast and slow layers for prediction
        combined_state = torch.cat([fast_nh2, slow_nh2], dim=-1)
        fused_state = F.relu(self.output_fusion(combined_state))
        
        # Pass through output layer
        x = self.flatten(fused_state)
        x = F.relu(x)
        x = self.fc(x)
        
        # Generate new state dictionary
        new_state_dict = {
            'fast': [fast_nh0.detach(), fast_nh1.detach(), fast_nh2.detach()],
            'slow': [slow_nh0.detach(), slow_nh1.detach(), slow_nh2.detach()],
            'step_count': step_count + 1
        }
        
        # Return output and updated state dictionary
        return F.log_softmax(x, dim=-1), new_state_dict 