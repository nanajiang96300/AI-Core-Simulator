
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MPNN(nn.Module):
    def __init__(self, N, M, hidden_size_1=16, hidden_size_2=8, u_size=8, L=2, output_size=4):
        super(MPNN, self).__init__()
        self.L = L
        self.u_size = u_size
        self.output_size = output_size # sqrt(Q) for PAM
        
        # Initialization Network (Eq 14)
        # Input: [y^T a_n, a_n^T a_n, sigma^2] -> 3 features
        self.init_mlp = nn.Linear(3, u_size)
        
        # Propagation MLP (Eq 16)
        # Input: [u_n, u_j, f_jn] -> 2*u_size + 2
        # Hidden: h1, h2
        # Output: u_size (message size, assumed same as u_size)
        self.prop_mlp = nn.Sequential(
            nn.Linear(2 * u_size + 2, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, u_size)
        )
        
        # Aggregation GRU (Eq 17)
        # Input to update: [sum(m_jn), d_n] -> u_size + 2
        # d_n = [r_n, Sigma_n]
        # GRU input size: u_size + 2
        # GRU hidden size: u_size (to match u_n size)
        # Paper says g_n is hidden state of GRU. u_n is updated from g_n via W2, b2.
        # But standard GRU updates hidden state directly.
        # "g_n = U(g_{n-1}, m_n)" -> U is GRU.
        # "u_n = W2 g_n + b2".
        # Let's assume GRU hidden size is h1 (as per paper text "g_n in R^Nh1").
        self.gru = nn.GRUCell(u_size + 2, hidden_size_1)
        self.update_mlp = nn.Linear(hidden_size_1, u_size) # W2, b2
        
        # Readout MLP (Eq 18)
        # Input: u_n -> u_size
        # Output: output_size (constellation size)
        self.readout_mlp = nn.Sequential(
            nn.Linear(u_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )
        
    def forward(self, y, A, sigma2, r, Sigma, u_prev=None, g_prev=None):
        # y: (B, M)
        # A: (B, M, N)
        # sigma2: (B, 1) or scalar
        # r: (B, N)
        # Sigma: (B, N)
        # u_prev: (B, N, u_size)
        
        B, M, N = A.shape
        device = A.device
        
        # 1. Edge Attributes f_jn = [a_n^T a_j, sigma2]
        # Compute Gram matrix G = A^T A (B, N, N)
        G = torch.bmm(A.transpose(1, 2), A) # (B, N, N)
        # f_jn_0 = G_jn.
        # f_jn_1 = sigma2.
        sigma2_expanded = sigma2.view(B, 1, 1).expand(B, N, N)
        # F_attr: (B, N, N, 2)
        F_attr = torch.stack([G, sigma2_expanded], dim=3)
        
        # 2. Initialization u^(0)
        # Eq 14: [y^T a_n, a_n^T a_n, sigma2]
        # y^T A = (A^T y)^T. 
        # ATy: (B, N) = bmm(A^T, y.unsqueeze(2)).squeeze(2)
        ATy = torch.bmm(A.transpose(1, 2), y.unsqueeze(2)).squeeze(2)
        # Diag of G is a_n^T a_n
        diag_G = torch.diagonal(G, dim1=1, dim2=2) # (B, N)
        
        sigma2_node = sigma2.view(B, 1).expand(B, N)
        
        # Init features: (B, N, 3)
        init_feat = torch.stack([ATy, diag_G, sigma2_node], dim=2)
        
        if u_prev is None:
            u = self.init_mlp(init_feat) # (B, N, u_size)
            g = torch.zeros(B * N, self.gru.hidden_size, device=device) # Init GRU hidden
        else:
            u = u_prev
            g = g_prev
            
        # 3. Message Passing Loop
        for l in range(self.L):
            # Propagation
            # Prepare input for MLP: [u_n, u_j, f_jn] for all j != n
            # We can do this with broadcasting
            # u_n: (B, N, 1, u) -> expand to (B, N, N, u)
            u_n_exp = u.unsqueeze(2).expand(B, N, N, self.u_size)
            # u_j: (B, 1, N, u) -> expand to (B, N, N, u)
            u_j_exp = u.unsqueeze(1).expand(B, N, N, self.u_size)
            
            # Concatenate: (B, N, N, 2*u + 2)
            prop_input = torch.cat([u_n_exp, u_j_exp, F_attr], dim=3)
            
            # Mask diagonal (j != n)
            mask = torch.eye(N, device=device).unsqueeze(0).unsqueeze(3) # (1, N, N, 1)
            # We can just zero out the diagonal messages or ignore them. 
            # The sum in aggregation should be over j != n.
            
            # Compute messages
            # (B, N, N, u_size)
            messages = self.prop_mlp(prop_input)
            
            # Zero out diagonal messages
            messages = messages * (1 - mask)
            
            # Aggregation
            # Sum over j (dim 2)
            m_sum = torch.sum(messages, dim=2) # (B, N, u_size)
            
            # Node attribute d_n = [r_n, Sigma_n]
            d_n = torch.stack([r, Sigma], dim=2) # (B, N, 2)
            
            # Input to GRU: [m_sum, d_n]
            gru_input = torch.cat([m_sum, d_n], dim=2) # (B, N, u_size + 2)
            
            # Flatten for GRUCell
            gru_input_flat = gru_input.view(B * N, -1)
            g = self.gru(gru_input_flat, g) # (B*N, hidden_size_1)
            
            # Update u
            u_flat = self.update_mlp(g) # (B*N, u_size)
            u = u_flat.view(B, N, self.u_size)
            
        # 4. Readout
        logits = self.readout_mlp(u) # (B, N, output_size)
        probs = F.softmax(logits, dim=2) # (B, N, output_size)
        
        return probs, u, g

class AMP_GNN(nn.Module):
    def __init__(self, M, N, T=10, L=2, const_values=None):
        super(AMP_GNN, self).__init__()
        self.M = M
        self.N = N # This is real-valued dimension (e.g. 64 for 32x32 complex)
        self.T = T
        self.L = L
        
        # MPNN Module
        # output_size depends on modulation (2 for QPSK/4QAM, 4 for 16QAM, 8 for 64QAM)
        if const_values is None:
            # Default to 16QAM (4-PAM)
            self.const_values = torch.tensor([-3.0, -1.0, 1.0, 3.0])
        else:
            self.const_values = const_values
            
        self.output_size = len(self.const_values)
        self.mpnn = MPNN(N, M, L=L, output_size=self.output_size)
        
    def forward(self, y, A, sigma2, x_true=None):
        # y: (B, M)
        # A: (B, M, N)
        # sigma2: (B, 1)
        
        B = y.shape[0]
        device = y.device
        
        # Initialization
        x_hat = torch.zeros(B, self.N, device=device)
        v_hat = torch.ones(B, self.N, device=device) * (self.N / self.M) # Paper says N/M, but for real? 
        # Check dimensionality. N/M is ratio.
        # Real system: (2N)/(2M) = N/M. So ratio is same.
        
        Z = torch.zeros(B, self.M, device=device) # Z^(-1) = 0? 
        # Algorithm 1: Z^(0) = y. Wait.
        # Step 2: Z^(0)_m = y_m. 
        # But Eq (8b) uses Z^(t-1).
        # Let's follow Algorithm 1 closely.
        # Init: x^(1)=0, v^(1)=N/M, Z^(0)=y.
        
        Z = y.clone()
        V = torch.zeros(B, self.M, device=device) # V^(0)? Not used in first iter 8b?
        # Eq 8b: ... - V^t (y - Z^(t-1)) / (sigma2 + V^(t-1))
        # Need V^(0).
        # Typically V^(0) is initialized large or using v^(1).
        # Let's trace loop t=1.
        # 8a: V^(1) uses v^(1). OK.
        # 8b: Z^(1) uses Z^(0) and V^(0). 
        # Paper doesn't specify V^(0).
        # Standard AMP: Onsager correction term.
        # Usually V^(0) is not needed if we start loop carefully.
        # Or V^(0) = sigma2 + something.
        # Let's assume V^(0) = 1 (or derived from v^(1)).
        # Actually, in first step, Z^(0)=y, so (y - Z^(0)) = 0. The correction term is 0.
        # So V^(0) value doesn't matter for t=1.
        
        V_old = torch.ones(B, self.M, device=device) # Placeholder
        
        # MPNN state
        u = None
        g = None
        
        final_x = x_hat
        
        for t in range(self.T):
            # 1. V^(t) (Eq 8a)
            # |a_mn|^2 * v_n
            A_sq = A ** 2
            V = torch.matmul(A_sq, v_hat.unsqueeze(2)).squeeze(2) # (B, M)
            
            # 2. Z^(t) (Eq 8b)
            # Term 1: A * x_hat
            Ax = torch.matmul(A, x_hat.unsqueeze(2)).squeeze(2)
            
            # Term 2: Onsager
            # (y - Z_old)
            resid = y - Z
            denom = sigma2.view(B, 1) + V_old
            factor = V / denom
            onsager = factor * resid
            
            Z_new = Ax - onsager
            
            # 3. Sigma^(t) (Eq 8c)
            # sum_m ( |a_mn|^2 / (sigma2 + V_m) )
            denom_new = sigma2.view(B, 1) + V
            inv_denom = 1.0 / denom_new
            Sigma_inv = torch.matmul(A_sq.transpose(1, 2), inv_denom.unsqueeze(2)).squeeze(2)
            Sigma = 1.0 / (Sigma_inv + 1e-8) # Avoid div by zero
            
            # 4. r^(t) (Eq 8d)
            # term: sum_m ( a*_mn (y - Z_new) / (sigma2 + V_m) )
            # A is real, a*_mn = a_mn.
            resid_new = y - Z_new
            scaled_resid = resid_new * inv_denom
            term2 = Sigma * torch.matmul(A.transpose(1, 2), scaled_resid.unsqueeze(2)).squeeze(2)
            r = x_hat + term2
            
            # 5. MPNN Denoising (Replaces 8e, 8f)
            # Input: y, A, sigma2, r, Sigma
            probs, u, g = self.mpnn(y, A, sigma2, r, Sigma, u, g)
            
            # Calculate mean and variance from probs
            # const_values: (Q_sqrt,) e.g. [-3, -1, 1, 3]
            # probs: (B, N, Q_sqrt)
            
            consts = self.const_values.to(device).view(1, 1, -1) # (1, 1, Q)
            
            # Mean: sum(p_i * s_i)
            x_hat_new = torch.sum(probs * consts, dim=2)
            
            # Variance: sum(p_i * |s_i|^2) - |mean|^2
            second_moment = torch.sum(probs * (consts ** 2), dim=2)
            v_hat_new = second_moment - (x_hat_new ** 2)
            
            # Update
            x_hat = x_hat_new
            v_hat = v_hat_new
            Z = Z_new
            V_old = V
            
        return x_hat

