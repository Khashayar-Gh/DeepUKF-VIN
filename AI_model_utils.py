import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import qMq
from concurrent.futures import ThreadPoolExecutor
torch.set_default_dtype(torch.float64)
from QNUKF import QNUKF
# Define the CNN model for images with scalar output
class vision_noise_model(nn.Module):
    def __init__(self):
        super(vision_noise_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(int(64 * (752/8) * (480/8)), 16)  # Adjust according to your input size
        self.fc2 = nn.Linear(16, 1)  # Scalar output

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), int(64 * (752/8) * (480/8)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RNNModelWithMasking(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModelWithMasking, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths = None):
        # packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # packed_output, _ = self.rnn(packed_input)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output, _ = self.rnn(x)
        # out = self.fc(output[torch.arange(output.size(0)), lengths - 1])
        out = self.fc(output)
        return out

class imu_noise_model(nn.Module):
    def __init__(self):
        super(imu_noise_model, self).__init__()
        self.fc1 = nn.Linear(6, 128)  # Adjust input size as needed
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 12)  # Scalar output

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define a custom loss function
class estimation_loss(nn.Module):
    def __init__(self):
        super(estimation_loss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.batch_norm = nn.BatchNorm1d(10)  # Adjust the feature size based on your input shape

    def forward(self, xhat, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() == 3:
            x = x[:, -1, :]

        # Apply batch normalization
        x = x[: , :10]
        xhat = xhat[: , :10]
        xhat = self.batch_norm(xhat)
        x = self.batch_norm(x)

        if xhat.size(0) > 150:
        # if False:
            ofset = 50
            q_e_batch = self.batch_process_with_loop(qMq, [xhat[ofset:, 0:4], x[ofset:, 0:4]])
            q_e = torch.mean(q_e_batch ** 2)
            p_e = self.loss_fn(xhat[ofset:, 4:7], x[ofset:, 4:7])
            v_e = self.loss_fn(xhat[ofset:, 7:10], x[ofset:, 7:10])
        else:
            q_e_batch = self.batch_process_with_loop(qMq, [xhat[:, 0:4], x[:, 0:4]])
            q_e = torch.mean(q_e_batch ** 2)
            p_e = self.loss_fn(xhat[:, 4:7], x[:, 4:7])
            v_e = self.loss_fn(xhat[:, 7:10], x[:, 7:10])

        # Scaling and adding a small constant
        epsilon = 1e-8
        q_e_scaled = q_e * 1000.0
        p_e_scaled = p_e * 600.0
        v_e_scaled = v_e * 100.0
        return q_e_scaled + p_e_scaled + v_e_scaled + epsilon
    
    @staticmethod
    def batch_process_with_loop(function, batch_data_list):
        """
        Applies a function to each element in a batch using a loop.
        
        Args:
            function (callable): A function to apply.
            batch_data (Tensor): A batch of data.
            
        Returns:
            Tensor: The result after applying the function to each element in the batch.
        """
        # tensor (batch_size, *shape)
        batch_size = batch_data_list[0].size(0)
        # Extract the first set of inputs to determine the output shape
        initial_inputs = [batch_data[0] for batch_data in batch_data_list]
        initial_output = function(*initial_inputs)
        
        # Preallocate the result tensor based on the initial output shape
        result_shape = (batch_size,) + initial_output.shape
        result = torch.zeros(result_shape, dtype=initial_output.dtype).cuda()
        result[0] = initial_output
        for i in range(1 , batch_size):
            inputs = [batch_data[i] for batch_data in batch_data_list]
        
            # Apply the function to the extracted elements
            result[i] = function(*inputs)

                # Define a function to process a single batch element and directly populate result
        # def process_element(i):
        #     inputs = [batch_data[i] for batch_data in batch_data_list]
        #     result[i] = function(*inputs)
        
        # # Use ThreadPoolExecutor for parallel processing
        # with ThreadPoolExecutor(max_workers=30) as executor:
        #     executor.map(process_element, range(1 , batch_size))
        
        return result
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class CNNGRUModel(nn.Module):
    def __init__(self, hidden_dim, num_layers, out_dim , dropout_prob):
        super(CNNGRUModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        # cnn_out_dim should be img ((dim / 16) * 32)
        self.gru = nn.GRU(int(16 * (752/16) * (480/16)) * 2, hidden_dim, num_layers,
                           batch_first=True , dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, left_x, right_x):
        batch_size, timesteps, C, H, W = left_x.size()

        # Process left images
        left_x = left_x.view(batch_size * timesteps, C, H, W)
        left_features = self.cnn(left_x)
        left_features = left_features.view(batch_size, timesteps, -1)

        # Process right images
        right_x = right_x.view(batch_size * timesteps, C, H, W)
        right_features = self.cnn(right_x)
        right_features = right_features.view(batch_size, timesteps, -1)

        # Concatenate features from left and right images
        combined_features = torch.cat((left_features, right_features), dim=2)
        combined_features = self.dropout(combined_features)

        # GRU
        gru_out, _ = self.gru(combined_features)
        out = self.fc(gru_out[:, -1, :])
        return out
    
class CNNModel(nn.Module):
    def __init__(self, out_dim, dropout_prob , hidden_dim=None):
        super(CNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        cnn_out_dim = int(16 * (752/16) * (480/16))  # Update this based on input image size
        if hidden_dim is None:
            self.fc = nn.Linear(cnn_out_dim * 2, out_dim)  # * 2 for left and right image concatenation
            self.fc_hid = None
        else:
            self.fc_hid = nn.Linear(cnn_out_dim * 2, hidden_dim)
            self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        

    def forward(self, left_x, right_x):
        batch_size, C, H, W = left_x.size()

        # Process left images
        left_features = self.cnn(left_x)
        left_features = left_features.view(batch_size, -1)

        # Process right images
        right_features = self.cnn(right_x)
        right_features = right_features.view(batch_size, -1)

        # Concatenate features from left and right images
        combined_features = torch.cat((left_features, right_features), dim=1)
        

        # Fully connected layer
        if self.fc_hid is None:
            combined_features = self.dropout(combined_features)
            out = self.fc(combined_features)
        else:
            hid = self.fc_hid(combined_features)
            hid = self.dropout(hid)
            out = self.fc(hid)
        return out
    
class CNNModel2(nn.Module):
    def __init__(self, out_dim, dropout_prob , hidden_dim=None):
        super(CNNModel2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        cnn_out_dim = int(16 * (752/16) * (480/16))  # Update this based on input image size
        if hidden_dim is None:
            self.fc = nn.Linear(cnn_out_dim * 2, out_dim)  # * 2 for left and right image concatenation
            self.fc_hid = None
        else:
            self.fc_hid = nn.Linear(cnn_out_dim * 2, hidden_dim)
            self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        

    def forward(self, left_x, right_x):
        batch_size, C, H, W = left_x.size()

        # Process left images
        left_features = self.cnn(left_x)
        left_features = left_features.view(batch_size, -1)

        # Process right images
        right_features = self.cnn(right_x)
        right_features = right_features.view(batch_size, -1)

        # Concatenate features from left and right images
        combined_features = torch.cat((left_features, right_features), dim=1)
        

        # Fully connected layer
        if self.fc_hid is None:
            combined_features = self.dropout(combined_features)
            out = self.fc(combined_features)
        else:
            hid = self.fc_hid(combined_features)
            hid = self.relu(hid)
            hid = self.dropout(hid)
            out = self.fc(hid)
        return out

class CNNModel3(nn.Module):
    def __init__(self, out_dim, dropout_prob, hidden_dim=None):
        super(CNNModel3, self).__init__()

        # Improved CNN with BatchNorm, Dropout, and increased channel depth
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # Increased channels from 8 to 16
            nn.BatchNorm2d(16),  # Added Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(0.2),  # Added Dropout for regularization
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  # Increased channels from 16 to 32
            nn.BatchNorm2d(32),  # Added Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(0.2)  # Added Dropout for regularization
        )

        # Calculate CNN output dimension based on input image size
        cnn_out_dim = int(32 * (752 / 16) * (480 / 16))  # Adjusted for updated output channels (32)

        # Fully connected layers
        if hidden_dim is None:
            self.fc = nn.Linear(cnn_out_dim * 2, out_dim)  # * 2 for concatenated left and right features
            self.fc_hid = None
        else:
            self.fc_hid = nn.Linear(cnn_out_dim * 2, hidden_dim)
            self.fc = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, left_x, right_x):
        batch_size, C, H, W = left_x.size()

        # Process left images
        left_features = self.cnn(left_x)
        left_features = left_features.view(batch_size, -1)

        # Process right images
        right_features = self.cnn(right_x)
        right_features = right_features.view(batch_size, -1)

        # Concatenate features from left and right images
        combined_features = torch.cat((left_features, right_features), dim=1)

        # Fully connected layers
        if self.fc_hid is None:
            combined_features = self.dropout(combined_features)
            out = self.fc(combined_features)
        else:
            hid = self.fc_hid(combined_features)
            hid = self.relu(hid)
            hid = self.dropout(hid)
            out = self.fc(hid)

        return out


class TimeSeriesGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, non_ts_input_dim=None):
        super(TimeSeriesGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc_gru = nn.Linear(hidden_dim, output_dim)

        # Handle non-time-series input if provided
        if non_ts_input_dim is not None:
            self.fc_non_ts = nn.Linear(non_ts_input_dim, hidden_dim)
            self.fc_combined = nn.Linear(hidden_dim * 2, output_dim)  # Combining GRU and non-TS inputs
            self.flatten = nn.Flatten()
        else:
            self.fc_combined = None

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x_ts, x_non_ts=None):
        gru_out, _ = self.gru(x_ts)
        gru_out = self.dropout(gru_out[..., -1, :])
        
        # If non-time-series input is provided, combine its output with GRU output
        if x_non_ts is not None:
            x_non_ts = self.flatten(x_non_ts)
            non_ts_output = self.fc_non_ts(x_non_ts)
            combined_output = torch.cat((gru_out, non_ts_output), dim=-1)
            output = self.fc_combined(combined_output)
        else:
            ts_output = self.fc_gru(gru_out)
            output = ts_output

        return output

class TimeSeriesGRU2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, non_ts_input_dim=None):
        super(TimeSeriesGRU2, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc_gru = nn.Linear(hidden_dim, output_dim)

        # Handle non-time-series input if provided
        if non_ts_input_dim is not None:
            self.fc_non_ts = nn.Linear(non_ts_input_dim, hidden_dim)
            self.fc_combined = nn.Linear(hidden_dim * 2, output_dim)  # Combining GRU and non-TS inputs
            self.flatten = nn.Flatten()
        else:
            self.fc_combined = None

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x_ts, x_non_ts=None):
        gru_out, _ = self.gru(x_ts)
        gru_out = self.dropout(gru_out[..., -1, :])
        
        # If non-time-series input is provided, combine its output with GRU output
        if x_non_ts is not None:
            x_non_ts = self.flatten(x_non_ts)
            non_ts_output = self.fc_non_ts(x_non_ts)
            combined_output = torch.cat((gru_out, non_ts_output), dim=-1)
            combined_output = self.relu(combined_output)
            output = self.fc_combined(combined_output)
        else:
            gru_out = self.relu(gru_out)
            ts_output = self.fc_gru(gru_out)
            output = ts_output

        return output
class TimeSeriesGRU3(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, non_ts_input_dim=None):
        super(TimeSeriesGRU3, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc_gru = nn.Linear(hidden_dim * 2, output_dim)  # Account for bidirectional GRU output

        if non_ts_input_dim is not None:
            self.fc_non_ts = nn.Sequential(
                nn.Linear(non_ts_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim)  # Deeper representation for non-TS input
            )
            self.fc_combined = nn.Linear(hidden_dim * 4, output_dim)  # *2 for bidirectional GRU, *2 for non-TS
            self.flatten = nn.Flatten()
        else:
            self.fc_combined = None

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x_ts, x_non_ts=None):
        gru_out, _ = self.gru(x_ts)
        gru_out = self.dropout(gru_out[..., -1, :])

        if x_non_ts is not None:
            x_non_ts = self.flatten(x_non_ts)
            non_ts_output = self.fc_non_ts(x_non_ts)
            combined_output = torch.cat((gru_out, non_ts_output), dim=-1)
            combined_output = self.relu(combined_output)
            output = self.fc_combined(combined_output)
        else:
            gru_out = self.relu(gru_out)
            output = self.fc_gru(gru_out)

        return output
    
class TimeSeriesGRU4(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob, non_ts_input_dim=None):
        super(TimeSeriesGRU4, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.fc_gru = nn.Linear(hidden_dim * 2, output_dim)  # Account for bidirectional GRU output

        if non_ts_input_dim is not None:
            self.fc_non_ts = nn.Sequential(
                nn.Linear(non_ts_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim)  # Deeper representation for non-TS input
            )
            self.fc_combined = nn.Linear(hidden_dim * 4, output_dim)  # *2 for bidirectional GRU, *2 for non-TS
            self.flatten = nn.Flatten()
        else:
            self.fc_combined = None

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

        # Attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),  # hidden_dim * 2 because of bidirectional GRU
            nn.Tanh(),
            nn.Softmax(dim=1)  # Normalize across time steps
        )

    def forward(self, x_ts, x_non_ts=None):
        # GRU output (batch_size, seq_length, hidden_dim*2)
        gru_out, _ = self.gru(x_ts)

        # Apply attention over the time dimension (seq_length)
        attention_weights = self.attention_layer(gru_out)  # (batch_size, seq_length, 1)
        attention_applied = torch.sum(attention_weights * gru_out, dim=1)  # Weighted sum over time steps (batch_size, hidden_dim*2)

        attention_applied = self.dropout(attention_applied)  # Apply dropout

        if x_non_ts is not None:
            x_non_ts = self.flatten(x_non_ts)
            non_ts_output = self.fc_non_ts(x_non_ts)

            # Concatenate GRU with non-time-series data
            combined_output = torch.cat((attention_applied, non_ts_output), dim=-1)
            combined_output = self.relu(combined_output)
            output = self.fc_combined(combined_output)
        else:
            attention_applied = self.relu(attention_applied)
            output = self.fc_gru(attention_applied)

        return output


    
class LossWrapper(nn.Module):
    def __init__(self, model, loss_fn):
        super(LossWrapper, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, imu, target, xhat_k_1 , P_k_1 , imu_noise_log_j , beta):
        tanh = nn.Tanh()
        imu_noise_std = self.model(imu)
        batch_size = imu.size(0)
        batch_size = 2
        x = torch.zeros(batch_size , 16).cuda()
        for i in range(batch_size):
            # print("i = ", i)
            QNUKF_ = QNUKF(xhat_k_1[i][-1] , P_k_1[i][-1])
            noise_vec_std = imu_noise_log_j.cuda()*10**(beta*tanh(imu_noise_std[i]))
            QNUKF_.predict(imu[i][-1][:3] , imu[i][-1][3:6] ,  noise_vec_std[0:3] ,
                            noise_vec_std[3:6] , noise_vec_std[6:9] ,
                            noise_vec_std[9:12])
            x[i] = QNUKF_.x #IMU train only
        loss = self.loss_fn(x, target)
        return loss

    