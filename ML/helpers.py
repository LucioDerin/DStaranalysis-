import torch
import uproot
from model import ParticleJetClassifier
from dataloader import ParticleJetDataset
import numpy as np

def model_from_ckpt(ckpt_path):
    model = ParticleJetClassifier()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def add_eval_to_file(input_root, ckpt_path, output_root=None, reduce_ds=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = model_from_ckpt(ckpt_path)
    model = model.to(device)

    # Load dataset
    dataset = ParticleJetDataset(input_root, reduce_ds=reduce_ds, evaluation=True)
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=5000, shuffle=False, collate_fn=lambda x: list(zip(*x)))

    # Prepare to store results
    part_output = []
    jet_output = []

    # Evaluate model on dataset
    for i,batch in enumerate(eval_dataloader):
        part_features, labels_particle, labels_jet, part_origins, part_versors, best_chi2s = batch
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
    
    # Save results to new ROOT file
    with uproot.recreate(output_root) as ofile:
        with uproot.open(input_root) as ifile:
            tree = ifile["tree"]
            num_events = int(reduce_ds) if reduce_ds >0 else tree.num_entries

            ofile.mktree("tree", {key: tree[key].array(library="np", entry_stop=num_events)[:reduce_ds] for key in tree.keys()})
        ofile.mktree("model_predictions", {"jet_output": jet_output,
                                           "part_output": part_output})

if __name__ == "__main__":
    
    ckpt_path = "ckpts/particle_jet_classifier.pth"
    input_root = "../data/data_py_top/output_Dijetcc_smeared_18164619.root"
    output_root = "../data/particle_jet_output.root"

    add_eval_to_file(input_root, ckpt_path, output_root=output_root, reduce_ds=0)
    print(f"Model evaluation completed. Results saved to {output_root}")
