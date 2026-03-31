
import torch
import numpy as np

def complex_to_real_matrix(H):
    # H: (batch, M, N) complex
    # Return: (batch, 2M, 2N) real
    real = H.real
    imag = H.imag
    # [ R  -I ]
    # [ I   R ]
    top = torch.cat([real, -imag], dim=2)
    bot = torch.cat([imag, real], dim=2)
    return torch.cat([top, bot], dim=1)

def complex_to_real_vector(x):
    # x: (batch, N) complex
    # Return: (batch, 2N) real
    return torch.cat([x.real, x.imag], dim=1)

def real_to_complex_vector(x):
    # x: (batch, 2N) real
    # Return: (batch, N) complex
    N = x.shape[1] // 2
    return torch.complex(x[:, :N], x[:, N:])

def generate_mimo_data(batch_size, M, N, snr_db, mod_order='16QAM'):
    # Generate H ~ CN(0, 1)
    H_real = torch.randn(batch_size, M, N) / np.sqrt(2)
    H_imag = torch.randn(batch_size, M, N) / np.sqrt(2)
    H = torch.complex(H_real, H_imag)
    
    # Generate x
    if mod_order == 'QPSK': # 4-QAM
        # Symbols: +/- 1 +/- 1j (normalized)
        # 2-PAM for real/imag
        pam_size = 2
        # Points: [-1, 1]
        points = torch.tensor([-1.0, 1.0])
        scale = 1.0 / np.sqrt(2)
    elif mod_order == '16QAM':
        # 4-PAM for real/imag
        pam_size = 4
        # Points: [-3, -1, 1, 3]
        points = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        scale = 1.0 / np.sqrt(10)
    elif mod_order == '64QAM':
        pam_size = 8
        points = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
        scale = 1.0 / np.sqrt(42)
    else:
        raise ValueError("Unsupported modulation")
        
    # Generate random indices for real and imag parts
    idx_real = torch.randint(0, pam_size, (batch_size, N))
    idx_imag = torch.randint(0, pam_size, (batch_size, N))
    
    x_real = points[idx_real] * scale
    x_imag = points[idx_imag] * scale
    x = torch.complex(x_real, x_imag)
    
    # Generate y = Hx + n
    # Signal power is 1 (due to scaling)
    # Noise power calculation
    # SNR = E[|Hx|^2] / E[|n|^2]
    # E[|Hx|^2] = N (since entries of H are variance 1, x is variance 1, sum of N terms) -> Wait.
    # Let's verify scaling.
    # H_ij ~ CN(0, 1). x_j ~ CN(0, 1).
    # (Hx)_i = sum_j H_ij x_j. Var = N * 1 * 1 = N.
    # So signal power per receive antenna is N.
    # Noise vector n ~ CN(0, sigma^2 I). Power per antenna is sigma^2.
    # SNR = N / sigma^2.
    # sigma^2 = N / 10^(SNR_dB/10)
    
    sigma2 = N / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(sigma2 / 2.0) # per real/imag dimension
    
    n_real = torch.randn(batch_size, M) * noise_std
    n_imag = torch.randn(batch_size, M) * noise_std
    n = torch.complex(n_real, n_imag)
    
    y = torch.matmul(H, x.unsqueeze(2)).squeeze(2) + n
    
    # Return real-valued equivalent
    H_r = complex_to_real_matrix(H)
    x_r = complex_to_real_vector(x)
    y_r = complex_to_real_vector(y)
    
    # Labels (indices) for SER calculation
    labels = torch.cat([idx_real, idx_imag], dim=1) # (batch, 2N)
    
    return y_r, H_r, x_r, labels, sigma2, points * scale

