import torch
import uproot
from model import ParticleJetClassifier
from dataloader import ParticleJetDataset
import numpy as np
import shutil

def model_from_ckpt(ckpt_path, Nfeature=8):
    model = ParticleJetClassifier(input_dim=Nfeature)
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
    dataset = ParticleJetDataset(input_root, reduce_ds=reduce_ds, evaluation=True, npad = 56)
    N = dataset.__len__()
    batch_size = 1024
    print(f"Dataset size: {N}, using batch size: {batch_size}")
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare to store results
    part_output = []
    jet_output = []
    jet_labels = []

    # Evaluate model on dataset
    for i,batch in enumerate(eval_dataloader):

        if i % 5 == 0:
            print(f"Processing batch {i+1}/{len(eval_dataloader)}")
        
        if vertexing:
            (part_features, labels_particle, labels_jet, part_origins, part_versors, best_chi2s) = batch
        else:
            (part_features, labels_particle, labels_jet) = batch

        part_features = part_features.to(device)          # [B, N, F]
        labels_particle = labels_particle.to(device)      # [B, N]

        padding_mask = (labels_particle == -1)

        pred_particle, pred_jet = model(part_features, padding_mask=padding_mask)
        pred_particle = torch.softmax(pred_particle, dim=-1)
        pred_jet = torch.sigmoid(pred_jet)

        part_output.append(pred_particle.detach().cpu().numpy())
        jet_output.append(pred_jet.detach().cpu().numpy())
        jet_labels.append(labels_jet.detach().cpu().numpy())

    part_output = np.concatenate(part_output, axis=0)
    jet_output = np.concatenate(jet_output, axis=0)
    jet_labels = np.concatenate(jet_labels, axis=0)

    shutil.copy(input_root, output_root)
    # Save results to new ROOT file
    with uproot.update(output_root) as ofile:
        ofile.mktree("model_predictions", {"jet_output": jet_output,
                                           "part_output": part_output,
                                           "jet_labels": jet_labels})

if __name__ == "__main__":
    
    ckpt_path = "ckpts/particle_jet_classifier_no_vtx.pth"
    input_root = "/mnt/project_mnt/atlas/atlas_gen_fs/lderin/DStaranalysis-/data/data_py_top/output_Dijetcc_smeared_18164628.root"
    key = input_root.split("/")[-1].split("_")[-1].replace(".root", "")
    output_root = f"../data/eval/particle_jet_output_{key}.root"

    add_eval_to_file(input_root, ckpt_path, output_root=output_root, reduce_ds=0)
    print(f"Model evaluation completed. Results saved to {output_root}")
