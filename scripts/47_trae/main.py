
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import generate_mimo_data
from model import AMP_GNN

def train(model, device, M, N, snr_db, mod_order, epochs=500, batch_size=64, samples=100000):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    # Generate Training Data
    print(f"Generating Training Data (M={M//2}, N={N//2}, {mod_order})...")
    y_train, H_train, x_train, _, sigma2_val, points = generate_mimo_data(
        samples, M // 2, N // 2, snr_db, mod_order=mod_order
    )
    
    # Create sigma2 tensor
    sigma2_train = torch.full((samples, 1), sigma2_val).float()
    
    dataset = torch.utils.data.TensorDataset(y_train, H_train, x_train, sigma2_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    print("Starting Training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for y_batch, H_batch, x_batch, sigma2_batch in dataloader:
            y_batch = y_batch.to(device)
            H_batch = H_batch.to(device)
            x_batch = x_batch.to(device)
            sigma2_batch = sigma2_batch.to(device)
            
            optimizer.zero_grad()
            
            x_hat = model(y_batch, H_batch, sigma2_batch)
            
            loss = loss_fn(x_hat, x_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
    return losses

def evaluate(model, device, M_complex, N_complex, snr_range, mod_order, batch_size=1000, samples=10000):
    model.eval()
    ser_list = []
    
    points = model.const_values.to(device)
    
    print(f"Starting AMP-GNN Evaluation ({M_complex}x{N_complex}, {mod_order})...")
    with torch.no_grad():
        for snr in snr_range:
            y_test, H_test, x_test, labels, sigma2_val, _ = generate_mimo_data(
                samples, M_complex, N_complex, snr, mod_order=mod_order
            )
            
            sigma2_test = torch.full((samples, 1), sigma2_val).float()
            
            dataset = torch.utils.data.TensorDataset(y_test, H_test, labels, sigma2_test)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            total_errors = 0
            total_symbols = 0
            
            for y_batch, H_batch, label_batch, sigma2_batch in dataloader:
                y_batch = y_batch.to(device)
                H_batch = H_batch.to(device)
                label_batch = label_batch.to(device)
                sigma2_batch = sigma2_batch.to(device)
                
                x_hat = model(y_batch, H_batch, sigma2_batch)
                
                # Demodulation / Hard Decision
                dist = torch.abs(x_hat.unsqueeze(2) - points.view(1, 1, -1))
                pred_idx = torch.argmin(dist, dim=2) 
                
                errors = (pred_idx != label_batch).sum().item()
                total_errors += errors
                total_symbols += label_batch.numel()
                
            ser = total_errors / total_symbols
            ser_list.append(ser)
            print(f"AMP-GNN SNR {snr} dB: SER = {ser:.6f}")
            
    return ser_list

def evaluate_mmse(device, M_complex, N_complex, snr_range, mod_order, batch_size=1000, samples=10000):
    ser_list = []
    
    # Get constellation points for demodulation
    _, _, _, _, _, points = generate_mimo_data(1, 1, 1, 10, mod_order=mod_order)
    points = points.to(device)
    
    print(f"Starting MMSE Evaluation ({M_complex}x{N_complex}, {mod_order})...")
    with torch.no_grad():
        for snr in snr_range:
            y_test, H_test, x_test, labels, sigma2_val, _ = generate_mimo_data(
                samples, M_complex, N_complex, snr, mod_order=mod_order
            )
            
            dataset = torch.utils.data.TensorDataset(y_test, H_test, labels)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
            
            total_errors = 0
            total_symbols = 0
            
            # MMSE Regularization term: sigma^2 * I
            # sigma2_val is total complex noise variance.
            lambda_reg = sigma2_val 
            
            for y_batch, H_batch, label_batch in dataloader:
                y_batch = y_batch.to(device)
                H_batch = H_batch.to(device)
                label_batch = label_batch.to(device)
                
                # MMSE Detection
                HTH = torch.bmm(H_batch.transpose(1, 2), H_batch) # (B, N, N)
                N_dim = HTH.shape[1]
                reg = torch.eye(N_dim, device=device).unsqueeze(0) * lambda_reg
                A_mat = HTH + reg
                HTy = torch.bmm(H_batch.transpose(1, 2), y_batch.unsqueeze(2)).squeeze(2) # (B, N)
                x_hat = torch.linalg.solve(A_mat, HTy)
                
                # Demodulation
                dist = torch.abs(x_hat.unsqueeze(2) - points.view(1, 1, -1))
                pred_idx = torch.argmin(dist, dim=2)
                
                errors = (pred_idx != label_batch).sum().item()
                total_errors += errors
                total_symbols += label_batch.numel()
                
            ser = total_errors / total_symbols
            ser_list.append(ser)
            print(f"MMSE SNR {snr} dB: SER = {ser:.6f}")
            
    return ser_list

def run_experiment(M_complex, N_complex, mod_order, device):
    print(f"\n=== Running Experiment: {M_complex}x{N_complex}, {mod_order} ===")
    
    M = 2 * M_complex
    N = 2 * N_complex
    snr_train = 20
    
    # Initialize Model
    _, _, _, _, _, points = generate_mimo_data(1, 1, 1, 10, mod_order=mod_order)
    model = AMP_GNN(M, N, T=10, L=2, const_values=points).to(device)
    
    # Model filename
    model_path = f"amp_gnn_model_{M_complex}x{N_complex}_{mod_order}.pth"
    
    # Train if needed (always retrain for new config)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training new model...")
        train_samples = 20000 # Reduced for speed in multi-run
        epochs = 20 # Reduced for speed
        losses = train(model, device, M, N, snr_train, mod_order, epochs=epochs, batch_size=64, samples=train_samples)
        
        # Save Loss Plot
        plt.figure()
        plt.plot(losses)
        plt.title(f"Training Loss ({M_complex}x{N_complex}, {mod_order})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.savefig(f"loss_{M_complex}x{N_complex}_{mod_order}.png")
        plt.close()
        
        torch.save(model.state_dict(), model_path)
    
    # Evaluate
    snr_range = range(0, 26, 5)
    samples_eval = 5000 # Reduced for speed
    
    ser_amp_gnn = evaluate(model, device, M_complex, N_complex, snr_range, mod_order, samples=samples_eval)
    ser_mmse = evaluate_mmse(device, M_complex, N_complex, snr_range, mod_order, samples=samples_eval)
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ser_amp_gnn, 'b-o', label='AMP-GNN')
    plt.semilogy(snr_range, ser_mmse, 'r-s', label='MMSE')
    plt.title(f"SER Comparison ({M_complex}x{N_complex}, {mod_order})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.grid(True, which="both", linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f"ser_{M_complex}x{N_complex}_{mod_order}.png")
    plt.close()
    
    return snr_range, ser_amp_gnn, ser_mmse

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    configs = [
        (32, 32, '16QAM'),
        (32, 32, 'QPSK'),
        (16, 16, '64QAM')
    ]
    
    for M, N, mod in configs:
        run_experiment(M, N, mod, device)

if __name__ == "__main__":
    main()
