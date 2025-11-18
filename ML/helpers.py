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
    # Load model
    model = model_from_ckpt(ckpt_path)

    # Load dataset
    dataset = ParticleJetDataset(input_root, reduce_ds=reduce_ds)

    # Prepare to store results
    part_output = []
    jet_output = []

    # Evaluate model on dataset
    for i in range(len(dataset)):
        data = dataset[i][0].unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output_part, output_jet = model(data)
            output_part = output_part.squeeze(0)  # Remove batch dimension
            output_jet = output_jet.squeeze(0)
            part_output.append(output_part.numpy())
            jet_output.append(output_jet.numpy())

    part_output = np.array(part_output, dtype=np.float32)
    jet_output = np.array(jet_output, dtype=np.float32).flatten()
    
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
    input_root = "../data/test.root"
    output_root = "../data/particle_jet_output.root"

    add_eval_to_file(input_root, ckpt_path, output_root=output_root, reduce_ds=10)
    print(f"Model evaluation completed. Results saved to {output_root}")
