import torch
import uproot
from model import ParticleJetClassifier
from dataloader import ParticleJetDataset
import numpy as np
import shutil

def model_from_ckpt(ckpt_path):
    model = ParticleJetClassifier()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def biggest_batch_size(N):
    if N <= 1:
        return N

    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            return N // i   # largest proper divisor

    return N  # N is prime


def add_eval_to_file(input_root, ckpt_path, output_root=None, reduce_ds=0, vertexing=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = model_from_ckpt(ckpt_path)
    model = model.to(device)

    # Load dataset
    dataset = ParticleJetDataset(input_root, reduce_ds=reduce_ds, evaluation=True)
    N = dataset.__len__()
    batch_size = biggest_batch_size(N)
    print(f"Dataset size: {N}, using batch size: {batch_size}")
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    # Prepare to store results
    part_output = []
    jet_output = []

    # Evaluate model on dataset
    for i,batch in enumerate(eval_dataloader):
        if vertexing:
            part_features, labels_particle, labels_jet, part_origins, part_versors, best_chi2s = batch
        else:
            part_features, labels_particle, labels_jet = batch
        part_features = [p.to(device) for p in part_features]

        if i%5==0:
            print(f"Evaluating batch {i}/{len(eval_dataloader)}")
        with torch.no_grad():
            for particles in part_features:
                particles = particles.unsqueeze(0)  # Add batch dimension
                output_part, output_jet = model(particles)
                output_part = output_part.squeeze(0)  # Remove batch dimension
                output_jet = output_jet.squeeze(0)
                part_output.append(output_part.cpu().numpy())
                jet_output.append(output_jet.cpu().numpy())

    part_output = np.array(part_output, dtype=np.float32)
    jet_output = np.array(jet_output, dtype=np.float32)
    
    shutil.copy(input_root, output_root)
    # Save results to new ROOT file
    with uproot.update(output_root) as ofile:
        ofile.mktree("model_predictions", {"jet_output": jet_output,
                                           "part_output": part_output})

if __name__ == "__main__":
    
    ckpt_path = "ckpts/particle_jet_classifier_no_vtx.pth"
    input_root = "../data/data_py_top/output_Dijetcc_smeared_18164619.root"
    output_root = "../data/particle_jet_output.root"

    add_eval_to_file(input_root, ckpt_path, output_root=output_root, reduce_ds=0)
    print(f"Model evaluation completed. Results saved to {output_root}")
