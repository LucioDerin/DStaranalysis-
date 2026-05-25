import torch
from torch.utils.data import Dataset
import uproot
import numpy as np

from vertexing import fit

class ParticleJetDataset(Dataset):
    def __init__(self, root_files, reduce_ds=0, Nfeatures = 10, evaluation = False, npad = None, fit_iter = 100):

        # Define Variables
        self.particle_variables = [
            "part_charge",
            "part_eta",
            "part_phi",
            "part_pt",
            "part_energy",
            "part_d0val",
            "part_d0err",
            "part_dzval",
            "part_dzerr",
            "part_mass", # --> truth variable, do not use as input
            "part_massReco",
            "part_pid", # --> truth variable, do not use as input
        ]
        self.particle_labels = ["part_isFromD", "part_isFromDStar"]
        self.jet_variables = ["jet_energy", "jet_eta"]

        self.Nfeatures = Nfeatures
        self.evaluation = evaluation
        self.npad = npad
        self.fit_iter = fit_iter

        self.read_events = 0
        
        self.full_data_array = {}

        # checking if the provided root_files is a list, if not, convert it to a list
        if isinstance(root_files, str):
            root_files = [root_files]

        for i, root_file in enumerate(root_files):

            self.tree = uproot.open(root_file)["tree"]
            # Get number of events
            if reduce_ds > 0:
                if self.read_events >= reduce_ds:
                    print(f"Reached event limit of {reduce_ds}. Stopping data loading.")
                    break
                self.to_read_events = reduce_ds - self.read_events
                self.nevents = min(self.to_read_events, self.tree.num_entries)
                self.read_events += self.nevents
            else:
                self.nevents = self.tree.num_entries

            print(f"Loading {self.nevents} events from file {root_file}...")

            if i == 0:
                # Load particle-level data
                for var in self.particle_variables + self.particle_labels:
                    self.full_data_array[var] = self.tree[var].array(library="np", entry_stop=self.nevents)

                # Load jet-level data
                for var in self.jet_variables:
                    self.full_data_array[var] = self.tree[var].array(library="np", entry_stop=self.nevents)
            else:
                # Load particle-level data
                for var in self.particle_variables + self.particle_labels:
                    new_data = self.tree[var].array(library="np", entry_stop=self.nevents)
                    self.full_data_array[var] = np.concatenate((self.full_data_array[var], new_data), axis=0)

                # Load jet-level data
                for var in self.jet_variables:
                    new_data = self.tree[var].array(library="np", entry_stop=self.nevents)
                    self.full_data_array[var] = np.concatenate((self.full_data_array[var], new_data), axis=0)

        # Building tracks' origins (ragged, before padding)
        origins = []
        for phi_jet, d0_jet, z0_jet in zip(
            self.full_data_array["part_phi"],
            self.full_data_array["part_d0val"],
            self.full_data_array["part_dzval"],
        ):
            origins_jet = []
            for phi, d0, z0 in zip(phi_jet, d0_jet, z0_jet):
                y0 = abs(d0) * np.sin(phi)
                x0 = abs(d0) * np.cos(phi)
                origins_jet.append([x0, y0, z0])
            origins.append(origins_jet)
        self.full_data_array["part_origin"] = np.array(origins, dtype=object)  # ragged

        # Building tracks' versors (ragged, before padding)
        versors = []
        for eta_jet, phi_jet in zip(
            self.full_data_array["part_eta"], self.full_data_array["part_phi"]
        ):
            versors_jet = []
            for eta, phi in zip(eta_jet, phi_jet):
                theta = 2 * np.arctan(np.exp(-eta))
                ax = np.sin(theta) * np.cos(phi)
                ay = np.sin(theta) * np.sin(phi)
                az = np.cos(theta)
                versors_jet.append([ax, ay, az])
            versors.append(versors_jet)
        self.full_data_array["part_versor"] = np.array(versors, dtype=object)  # ragged

        # Best chi2 — computed before padding, on true vertex-associated tracks only
        best_chi2s = []
        fitted_vtxs = []
        for o, a, label in zip(
            self.full_data_array["part_origin"],
            self.full_data_array["part_versor"],
            self.full_data_array["part_isFromD"],
        ):
            o_arr = np.array(o)   # [n, 3]
            a_arr = np.array(a)   # [n, 3]
            fit_o = torch.tensor(o_arr[label == 1]).float()
            fit_a = torch.tensor(a_arr[label == 1]).float()

            best_chi2, fv = fit(fit_o, fit_a, None, self.fit_iter)
            best_chi2s.append(best_chi2.item())
            fitted_vtxs.append(fv.detach().numpy())  # [3]

        self.full_data_array["best_chi2"] = np.array(best_chi2s, dtype=np.float32)   # [nevents]
        self.full_data_array["fitted_vtx"] = np.array(fitted_vtxs, dtype=np.float32) # [nevents, 3]

        # Pad tracks (including origins and versors)
        self.__pad_tracks()

        # Normalize data
        self.__normalize_full_data__()

        # Define jet-level label (c-jet: at least one D* particle)
        self.jet_isCJet = np.array([
            1 if bool(np.any(self.full_data_array["part_isFromD"][idx] > 0)) else 0
            for idx in range(len(self.full_data_array["part_isFromD"]))
        ], dtype=np.float32)

    def __len__(self):
        return len(self.full_data_array["jet_energy"])
    
    def __pad_tracks(self):

        if self.npad is None:
            self.npad = max(
                len(self.full_data_array["part_eta"][i]) for i in range(len(self.full_data_array["jet_energy"]))
            )

        print(f"Padding all events to {self.npad} tracks.")

        for key in self.particle_variables:
            padded_array = []
            for i in range(len(self.full_data_array["jet_energy"])):
                current_array = self.full_data_array[key][i]
                pad_size = self.npad - len(current_array)
                if pad_size > 0:
                    if isinstance(current_array[0], np.ndarray):
                        pad_shape = (pad_size,) + current_array[0].shape
                        padding = np.zeros(pad_shape)
                    else:
                        padding = np.zeros(pad_size)
                    padded_array.append(np.concatenate([current_array, padding], axis=0))
                else:
                    padded_array.append(current_array[:self.npad])
            self.full_data_array[key] = np.array(padded_array, dtype=np.float32)

        for key in self.particle_labels:
            padded_array = []
            for i in range(len(self.full_data_array["jet_energy"])):
                current_array = self.full_data_array[key][i]
                pad_size = self.npad - len(current_array)
                if pad_size > 0:
                    padding = -1 * np.ones(pad_size)  # Use -1 for padding labels
                    padded_array.append(np.concatenate([current_array, padding], axis=0))
                else:
                    padded_array.append(current_array[:self.npad])
            self.full_data_array[key] = np.array(padded_array, dtype=np.float32)

        # Pad origins and versors to [nevents, npad, 3]
        for key in ["part_origin", "part_versor"]:
            padded_array = np.zeros((len(self.full_data_array[key]), self.npad, 3), dtype=np.float32)
            for i, jet in enumerate(self.full_data_array[key]):
                jet_arr = np.array(jet, dtype=np.float32)  # [n, 3]
                n = min(len(jet_arr), self.npad)
                padded_array[i, :n, :] = jet_arr[:n, :]
            self.full_data_array[key] = padded_array  # [nevents, npad, 3]

    def __normalize_full_data__(self):
    
        # Particle-level variables (2D: events × tracks)
        particle_mask = self.full_data_array["part_isFromD"] != -1
    
        for key in self.particle_variables:
            if key in ["part_charge", "part_pid", "part_mass"]:
                continue
            
            arr = self.full_data_array[key]
    
            mean = np.mean(arr[particle_mask])
            std  = np.std(arr[particle_mask]) + 1e-6
    
            self.full_data_array[key] = (arr - mean) / std

        for key in self.particle_variables:
            self.full_data_array[key][particle_mask == False] = 0.0

        # Jet-level variables (1D: events)
        for key in self.jet_variables:
            arr = self.full_data_array[key]
            mean = np.mean(arr)
            std  = np.std(arr) + 1e-6
            self.full_data_array[key] = (arr - mean) / std

    def __getitem__(self, idx):

        # Extract per-particle features
        part_features = np.stack([
            self.full_data_array["part_charge"][idx],
            self.full_data_array["part_eta"][idx],
            self.full_data_array["part_phi"][idx],
            self.full_data_array["part_pt"][idx],
            self.full_data_array["part_d0val"][idx],
            self.full_data_array["part_d0err"][idx],
            self.full_data_array["part_dzval"][idx],
            self.full_data_array["part_dzerr"][idx],
        ], axis=1)

        part_features = part_features[:, :self.Nfeatures]  # Select only the first Nfeatures

        # Convert one-hot labels to three-class indices
        isFromD = self.full_data_array["part_isFromD"][idx]
        isFromDStar = self.full_data_array["part_isFromDStar"][idx]

        # Class 2: D* meson
        # Class 1: D meson
        # Class 0: Everything else (background)
        labels_particle = np.full_like(isFromD, 0)

        valid_mask = isFromD != -1

        labels_particle[valid_mask & (isFromD > 0)] = 1
        labels_particle[valid_mask & (isFromDStar > 0) & (isFromD == 0)] = 2
        labels_particle[~valid_mask] = -1

        # Extract jet-level label
        label_jet = self.jet_isCJet[idx]

        return (
            torch.tensor(part_features, dtype=torch.float32),                               # [npad, F]
            torch.tensor(labels_particle, dtype=torch.float32),                              # [npad]
            torch.tensor(label_jet, dtype=torch.float32),                                    # scalar
            torch.tensor(self.full_data_array["part_origin"][idx], dtype=torch.float32),     # [npad, 3]
            torch.tensor(self.full_data_array["part_versor"][idx], dtype=torch.float32),     # [npad, 3]
            torch.tensor(self.full_data_array["fitted_vtx"][idx], dtype=torch.float32),      # [3]
            torch.tensor(self.full_data_array["best_chi2"][idx], dtype=torch.float32),       # scalar
        )
    
    def get_particles_loss_class_weights(self):
        all_labels_fromD = self.full_data_array["part_isFromD"].flatten()
        all_labels_fromDStar = self.full_data_array["part_isFromDStar"].flatten()
        all_labels = np.full_like(all_labels_fromD, 0)
        valid_mask = all_labels_fromD != -1
        all_labels[valid_mask & (all_labels_fromD > 0)] = 1
        all_labels[valid_mask & (all_labels_fromDStar > 0) & (all_labels_fromD == 0)] = 2
        all_labels = all_labels[valid_mask]

        class_counts = np.bincount(all_labels.astype(int))
        total_samples = len(all_labels)
        class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
        return torch.tensor(class_weights, dtype=torch.float32)

    def get_jets_loss_class_pos_weight(self):
        all_labels = self.jet_isCJet
        pos_weight = (len(all_labels) - np.sum(all_labels)) / (np.sum(all_labels) + 1e-6)
        return torch.tensor(pos_weight, dtype=torch.float32)

    def get_npad(self):
        return self.npad

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    chi2s = []
    labels_jets = []
    root_files = ["/home/lucio/1_FW_Areas/gitrepos/github/PhD_repos/DStaranalysis-/data/output_Dijetcc_smeared_18164619.root"]
    dataset = ParticleJetDataset(root_files=root_files, reduce_ds=10000)
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        part_features, labels_particle, label_jet, origins, versors, fitted_vtxs, best_chi2s = dataset[i]
        #print(f"Particle features shape: {part_features.shape}")
        #print(f"Particle labels shape: {labels_particle.shape}")
        #print(f"Jet label: {label_jet}")
        #print(f"Origins shape: {origins.shape}")
        #print(f"Versors shape: {versors.shape}")
        #print(f"Fitted vertex shape: {fitted_vtxs.shape}")
        #print(f"Best chi2: {best_chi2s}")

        chi2s.append(best_chi2s.item())
        labels_jets.append(label_jet.item())
        #print("-" * 50)


    plt.subplot(121)
    plt.hist(chi2s, bins=50)
    plt.semilogy()
    plt.xlabel("Best chi2")
    plt.ylabel("Frequency")
    plt.title("All jets")

    chi2s = np.array(chi2s)
    labels_jets = np.array(labels_jets)
    plt.subplot(122)
    plt.hist(chi2s[labels_jets == 1], bins=50, alpha=0.5)
    plt.semilogy()
    plt.xlabel("Best chi2")
    plt.ylabel("Frequency")
    plt.title("D* jets")

    plt.suptitle("Distribution of Best chi2 for True Vertex-Associated Tracks")
    plt.tight_layout()
    plt.savefig("best_chi2_distribution.png")