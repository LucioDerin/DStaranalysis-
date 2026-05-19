import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import ParticleJetDataset  # Import dataset
from model import ParticleJetClassifier  # Import model
import sys
import time as t

# Load Dataset and Dataloader
Nfeature = 8
root_file_prefix = "/mnt/project_mnt/atlas/atlas_gen_fs/lderin/DStaranalysis-/data/data_py_top/output_Dijetcc_smeared_181646"  # Change this to your actual ROOT file
root_files = []
for i in range(19, 26):
    root_files.append(f"{root_file_prefix}{i}.root")
dataset = ParticleJetDataset(root_files, reduce_ds=-1, Nfeatures=Nfeature)
npad = dataset.get_npad()
val_dataset = ParticleJetDataset([f"{root_file_prefix}26.root"], reduce_ds=10000, Nfeatures=Nfeature, npad = npad)

batch_size = 1024

print(f"Dataset size: {len(dataset)}, using batch size: {batch_size}")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = ParticleJetClassifier(input_dim=Nfeature).to(device)

# Losses
class_weights_particle = dataset.get_particles_loss_class_weights().to(device)
criterion_particle = nn.CrossEntropyLoss(weight=class_weights_particle, reduction="mean")
pos_weight_jet = dataset.get_jets_loss_class_pos_weight().to(device)
criterion_jet = nn.BCEWithLogitsLoss(pos_weight=pos_weight_jet, reduction="mean")

print("\n***** INFO *****")
print(f"Particle class weights: {class_weights_particle.cpu().numpy()}")
print(f"Jet class weights: {pos_weight_jet.cpu().numpy()}")
print("****************\n")

# **Define Optimizer**
initial_lr = 1e-3  # Starting learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# **Adaptive Learning Rate Scheduler**
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
num_epochs = 500

best_model = None
best_val_loss = float('inf')
best_epoch = -1

for epoch in range(num_epochs):
    start_epoch_time = t.time()
    total_loss = 0.0
    particles_loss = 0.0
    jets_loss = 0.0
    model.train()

    for (part_features, labels_particle, labels_jet) in dataloader:

        part_features = part_features.to(device)          # [B, N, F]
        labels_particle = labels_particle.to(device)      # [B, N]
        labels_jet = labels_jet.to(device).float()        # [B] or [B, 1]


        optimizer.zero_grad()

        # ----------------------------------
        # Particle-level loss (mask padding)
        # ----------------------------------
        padding_mask = labels_particle == -1   # shape [B, N]

        # Forward pass (vectorized)
        pred_particle, pred_jet = model(part_features, padding_mask)

        # Flatten only valid entries
        # supporting track-less jets
        if padding_mask.any():
            particle_loss = criterion_particle(
                pred_particle[~padding_mask],
                labels_particle[~padding_mask].long()
            )
        else:
            particle_loss = torch.tensor(0.0, device=device)

        # ------------------
        # Jet-level loss
        # ------------------
        jet_loss = criterion_jet(
            pred_jet.flatten(),
            labels_jet.flatten()
        )

        #print(f"Particle Loss: {particle_loss.item():.4f}, Jet Loss: {jet_loss.item():.4f}")
        if particle_loss.isnan():
            with open("faulty_batch.log", "a") as f:
                f.write(f"Epoch {epoch+1}, Particle Loss is NaN. Batch details:\n")
                for i in range(part_features.size(0)):
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  Features: {part_features[i].cpu().numpy()}\n")
                    f.write(f"  Particle Labels: {labels_particle[i].cpu().numpy()}\n")
                    f.write(f"  Jet Label: {labels_jet[i].item()}\n")
            sys.exit(1)

        # Combined loss
        batch_loss = (0.3 * particle_loss + jet_loss)
        particles_loss += particle_loss.item()
        jets_loss += jet_loss.item()


        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()

    total_loss = total_loss / len(dataloader)
    particles_loss = particles_loss / len(dataloader)
    jets_loss = jets_loss / len(dataloader)

    scheduler.step()  # only if this is a per-batch scheduler

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_particles_loss = 0.0
    val_jets_loss = 0.0
    with torch.no_grad():
        for (val_part_features, val_labels_particle, val_labels_jet) in val_dataloader:
            val_part_features = val_part_features.to(device)
            val_labels_particle = val_labels_particle.to(device)
            val_labels_jet = val_labels_jet.to(device).float()
            val_padding_mask = val_labels_particle == -1

            # Forward pass
            val_pred_particle, val_pred_jet = model(val_part_features, val_padding_mask)

            # Compute validation loss
            if val_padding_mask.any():
                val_particle_loss = criterion_particle(
                    val_pred_particle[~val_padding_mask],
                    val_labels_particle[~val_padding_mask].long()
                )
            else:
                val_particle_loss = torch.tensor(0.0, device=device)

            val_jet_loss = criterion_jet(
                val_pred_jet.flatten(),
                val_labels_jet.flatten()
            )

            batch_val_loss = 0.3 * val_particle_loss + val_jet_loss
            val_particles_loss += val_particle_loss.item()
            val_jets_loss += val_jet_loss.item()
            val_loss += batch_val_loss.item()
    val_loss = val_loss / len(val_dataloader)
    val_particles_loss = val_particles_loss / len(val_dataloader)
    val_jets_loss = val_jets_loss / len(val_dataloader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()  # Save the best model weights
        best_epoch = epoch + 1

    lr = scheduler.get_last_lr()[0]
    end_epoch_time = t.time()
    epoch_duration = end_epoch_time - start_epoch_time
    remaining_time = epoch_duration * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}, ETA: {int(remaining_time//3600)}h{int((remaining_time%3600)//60)}m")
    with open("logs/train_no_vtx.log", "a") as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Train Particles Loss: {particles_loss:.4f}, Train Jets Loss: {jets_loss:.4f}, Val Loss: {val_loss:.4f}, Val Particles Loss: {val_particles_loss:.4f}, Val Jets Loss: {val_jets_loss:.4f}, LR: {lr:.6f}\n")


print("Training complete!")

# Save the best model weights
torch.save(best_model, f"ckpts/particle_jet_classifier_no_vtx_best_epoch_{best_epoch}.pth")