import numpy as np
import torch
import torch.nn.functional as F   


def preprocess_inference_data(x, batched, batch_size, sequence_length, sample_rate):
    # Reshapes inference signal into batch for batched inference with 5 ms overlap between folds
    if not batched or len(x) <= sequence_length:
        return [x]
    else:
        overlap = int(0.005 * sample_rate)  # 5ms overlap between folds
        hop_size = int(sequence_length - overlap)
        num_folds = int(1 + np.ceil((len(x) - sequence_length) / hop_size))
        pad_len = int((sequence_length + (num_folds - 1) * hop_size) - len(x))
        x = F.pad(x, (0, pad_len))
        folds = [x[i * hop_size:i * hop_size + sequence_length] for i in range(num_folds)]
        return [torch.stack(folds[i:i + batch_size]) for i in range(0, len(folds), batch_size)]


def postprocess_inference_data(y, batched, sample_rate):
    # Reshapes batch into output signal
    if not batched:
        return y[0]
    else:
        y = torch.cat(y, dim=0)
        sequence_length = y.shape[1]
        overlap = int(0.005 * sample_rate)  # 5ms overlap between folds
        hop_size = int(sequence_length - overlap)
        t = np.linspace(-1, 1, overlap, dtype=np.float64)
        fade_in = torch.tensor(np.sqrt(0.5 * (1 + t))).to(y.device)
        fade_out = torch.tensor(np.sqrt(0.5 * (1 - t))).to(y.device)
        y[1:, :overlap] *= fade_in
        y[:-1, -overlap:] *= fade_out
        unfolded = torch.zeros(sequence_length + (y.shape[0] - 1) * hop_size).to(y.device)
        for i in range(y.shape[0]):
            start = i * hop_size
            unfolded[start:start + sequence_length] += y[i]
        return unfolded
