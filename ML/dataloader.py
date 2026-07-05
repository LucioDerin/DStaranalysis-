import torch
from torch.utils.data import Dataset
import uproot
import numpy as np

class ParticleJetDataset(Dataset):
    def __init__(self, root_files, reduce_ds=0, Nfeatures = -1, evaluation = False, npad = None):
        
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
        self.particle_labels = ["part_isFromD0", "part_isFromDp","part_isFromDs", "part_isFromCB", "part_isFromDStar"]
        self.jet_variables = ["jet_energy", "jet_eta", "jet_pt"]

        self.Nfeatures = Nfeatures
        self.evaluation = evaluation
        self.npad = npad

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

        # Convert uproot output to correct format
        # for key in self.full_data_array.keys():
        #     if isinstance(self.full_data_array[key], dict):
        #         self.full_data_array[key] = list(self.full_data_array[key].values())
        #     else:
        #         self.full_data_array[key] = np.array(self.full_data_array[key])

        # Pad tracks
        self.__pad_tracks()

        # Normalize data
        self.__normalize_full_data__()

        # Define jet-level label (c-jet: at least one D* particle)
        self.jet_isD0Jet = np.array([
            1 if bool(np.any(self.full_data_array["part_isFromD0"][idx] > 0)) else 0
            for idx in range(len(self.full_data_array["part_isFromD0"]))
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
                    padded_array.append(current_array)
            
            self.full_data_array[key] = np.array(padded_array)

        for key in self.particle_labels:
            padded_array = []
            for i in range(len(self.full_data_array["jet_energy"])):
                current_array = self.full_data_array[key][i]
                pad_size = self.npad - len(current_array)
                if pad_size > 0:
                    padding = -1 * np.ones(pad_size)  # Use -1 for padding labels
                    padded_array.append(np.concatenate([current_array, padding], axis=0))
                else:
                    padded_array.append(current_array)
            
            self.full_data_array[key] = np.array(padded_array)

    def __normalize_full_data__(self):
    
        # Particle-level variables (2D: events × tracks)
        particle_mask = self.full_data_array["part_isFromD0"] != -1
    
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
            #if key in ["jet_pt", "jet_eta"]:
            #    continue
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
            #self.full_data_array["part_energy"][idx],
            #self.full_data_array["part_massReco"][idx],
            # target label -> Remove before training !!!
            self.full_data_array["part_isFromD0"][idx],
            self.full_data_array["part_isFromDp"][idx],
            self.full_data_array["part_isFromDs"][idx],
            self.full_data_array["part_isFromCB"][idx],
            self.full_data_array["part_isFromDStar"][idx],
        ], axis=1)

        if self.Nfeatures > 0:
            part_features = part_features[:, :self.Nfeatures]  # Select only the first Nfeatures

        # **Convert one-hot labels to three-class indices**
        isFromD0 = self.full_data_array["part_isFromD0"][idx]
        #isFromDStar = self.full_data_array["part_isFromDStar"][idx]

        # Convert to class index:
        # Class 2: D* meson
        # Class 1: D meson
        # Class 0: Everything else (background)
        labels_particle = np.full_like(isFromD0, 0)

        valid_mask = isFromD0 != -1

        labels_particle[valid_mask & (isFromD0 > 0)] = 1
        #labels_particle[valid_mask & (isFromDStar > 0) & (isFromD0 == 0)] = 2

        labels_particle[~valid_mask] = -1

        # Extract jet-level label
        label_jet = self.jet_isD0Jet[idx]

        return (torch.tensor(part_features, dtype=torch.float32),
                torch.tensor(labels_particle, dtype=torch.long),
                torch.tensor(label_jet, dtype=torch.float32))
    
    def get_particles_loss_class_weights(self):
        all_labels_fromD = self.full_data_array["part_isFromD0"].flatten()
        #all_labels_fromDStar = self.full_data_array["part_isFromDStar"].flatten()
        all_labels = np.full_like(all_labels_fromD, 0)
        valid_mask = all_labels_fromD != -1
        all_labels[valid_mask & (all_labels_fromD > 0)] = 1
        #all_labels[valid_mask & (all_labels_fromDStar > 0) & (all_labels_fromD == 0)] = 2
        all_labels = all_labels[valid_mask]

        class_counts = np.bincount(all_labels.astype(int))
        total_samples = len(all_labels)
        class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
        return torch.tensor(class_weights, dtype=torch.float32)

    def get_jets_loss_class_pos_weight(self):
        all_labels = self.jet_isD0Jet
        pos_weight = (len(all_labels) - np.sum(all_labels)) / (np.sum(all_labels) + 1e-6)
        return torch.tensor(pos_weight, dtype=torch.float32)

    def get_npad(self):
        return self.npad

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dataset = ParticleJetDataset(root_files="../data/test_50Smear.root")
    print(f"Dataset length: {len(dataset)}")

    label_jets = []
    label_particles = []
    features_particles = []
    charges = []

    for i in range(len(dataset)):
        part_features, labels_particle, label_jet = dataset[i]
        
        label_jets.append(label_jet.item())
        label_particles.append(labels_particle.numpy())
        charges.append(part_features[:, 0].numpy())  # Assuming charge is the first feature
        features_particles.append(part_features.numpy())

    label_jets = np.array(label_jets)

    # Plotting jet label
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(label_jets, bins=2, alpha=0.7, color='blue')
    print(f"Number of D0 jets: {np.sum(label_jets == 1)}, Number of non-D0 jets: {np.sum(label_jets == 0)}, ratio: {np.sum(label_jets == 1) / (np.sum(label_jets == 0) + 1e-6):.4f}")
    plt.xlabel(f'Jet Label (0: non-D0 jet, 1: D0 jet), ratio: {np.sum(label_jets == 1) / (np.sum(label_jets == 0) + 1e-6):.4f}')
    plt.ylabel('Count')
    plt.title('Distribution of Jet Labels') 

    # plotting particles label
    plt.subplot(1, 2, 2)
    label_names = [
        "From D0",
        "From Dp",
        "From Ds",
        "From CB",
        "From D*"
    ]
    counts = []
    for i, label_name in enumerate(label_names):
        count = np.sum([np.sum(part_features[:,8+i] == 1) for part_features in features_particles])  # Count particles with this label
        counts.append(count)

    counts = np.array(counts)
    total_particles = np.sum(counts)
    for label_name, count in zip(label_names, counts):
        plt.bar(label_name, count, alpha=0.7, label=f"{label_name}: {(count/total_particles)*100:.2f}%")
    
    plt.xlabel('Particle Label')
    plt.ylabel('Count')
    plt.title('Distribution of Particle Labels')
    plt.legend()
    plt.tight_layout()
    plt.savefig("label_distributions.png")


    # Charge reco

    # Summing D0 particles charges for each jet to see if they add up to zero
    sum_charges = []
    for j in range(len(label_jets)):
        if label_jets[j] == 1:  # Only consider jets labeled as D0 jets
            d0_mask = label_particles[j] == 1  # Mask for D0 particles
            sum_charge = np.sum(charges[j][d0_mask])  # Sum charges of D0 particles in the jet
            sum_charges.append(sum_charge)

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.hist(sum_charges, bins=20, alpha=0.7, color='orange')
    plt.xlabel('Sum of Charges of D0 Particles in Jet')
    plt.ylabel('Count')
    plt.title('Distribution of Sum of Charges\nfor D0 Particles in Jets')

    plt.subplot(1, 2, 2)

    # Summing Dp particles charges for each jet to see if they add up to +1
    sum_charges_dp = []
    for j in range(len(label_jets)):
        dp_label = features_particles[j][:, 9]  # Assuming Dp label is at index 9
        dp_mask = dp_label == 1  # Mask for Dp particles
        sum_charge_dp = np.sum(charges[j][dp_mask])  # Sum charges of Dp particles in the jet
        sum_charges_dp.append(sum_charge_dp)

    plt.hist(sum_charges_dp, bins=20, alpha=0.7, color='green')
    plt.xlabel('Sum of Charges of Dp Particles in Jet')
    plt.ylabel('Count')
    plt.title('Distribution of Sum of Charges\nfor Dp Particles in Jets')
    plt.tight_layout()
    plt.savefig("charge_distributions.png")

    # Jet label split by Dwhatever particle presence
    label_names.append("Other")
    labels_jet_allDtypes = []
    for i, label_name in enumerate(label_names):
        if label_name == "Other":
            current_type_labels = []
            for j in range(len(label_jets)):
                features = features_particles[j]
                if np.any(features[:, 8:8+len(label_names)-1] == 1):  # Check if there's at least one particle of any type in the jet
                    current_type_labels.append(0)  
                else:
                    current_type_labels.append(1)
            labels_jet_allDtypes.append(current_type_labels)
        else:
            current_type_labels = []
            for j in range(len(label_jets)):
                features = features_particles[j]
                if np.any(features[:, 8+i] == 1):  # Check if there's at least one particle of this type in the jet
                    current_type_labels.append(1)  
                else:
                    current_type_labels.append(0)
            labels_jet_allDtypes.append(current_type_labels)

    labels_jet_allDtypes = np.array(labels_jet_allDtypes)

    # Plotting the number of jets with at least one particle of each type
    plt.figure(figsize=(10, 6))
    for i, label_name in enumerate(label_names):
        count = np.sum(labels_jet_allDtypes[i])
        plt.bar(label_name, count, alpha=0.7, label=f"{label_name}: {(count/len(label_jets))*100:.2f}%")

    plt.xlabel('Particle Type')
    plt.ylabel('Number of Jets with at least one particle of this type')
    total_jets = len(label_jets)
    total_counts = np.sum([np.sum(labels_jet_allDtypes[i]) for i in range(len(label_names))])
    plt.title(f'Number of Jets with at least one particle of each type\nTotal counts over number of jets: {total_counts}/{total_jets} ({(total_counts/total_jets)*100:.2f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("jet_particle_type_distribution.png")

    # Pt and eta distributions of jets by particle type
    plt.figure(figsize=(12, 6))
    jet_pts_by_type = {label_name: [] for label_name in label_names}
    jet_etas_by_type = {label_name: [] for label_name in label_names}
    for j in range(len(label_jets)):
        jet_pt = dataset.full_data_array["jet_pt"][j]
        jet_eta = dataset.full_data_array["jet_eta"][j]
        features = features_particles[j]
        for i, label_name in enumerate(label_names[:-1]):
            if np.any(features[:, 8+i] == 1):  # Check if there's at least one particle of this type in the jet
                jet_pts_by_type[label_name].append(jet_pt)
                jet_etas_by_type[label_name].append(jet_eta)
                break
            jet_pts_by_type["Other"].append(jet_pt)
            jet_etas_by_type["Other"].append(jet_eta)
    xrange = (0, 150) 
    Nbins = 20
    cumulative_pts = []
    cumulative_etas = []
    for label_name in label_names[:-1]:
        cumulative_pts += jet_pts_by_type[label_name]
        cumulative_etas += jet_etas_by_type[label_name]
    plt.hist(cumulative_pts, bins=Nbins, alpha=0.5, label='All c-jets', color='orange', range=xrange, density=True, histtype='step')
    plt.hist(jet_pts_by_type["Other"], bins=Nbins, alpha=0.7, label='Other jets', color='blue', range=xrange, density=True, histtype='step')

    plt.xlabel('Jet Pt')
    plt.ylabel('Count')
    plt.semilogy()
    plt.xlim(xrange)
    plt.title('Distribution of Jet Pt by Particle Type')
    plt.legend()
    plt.tight_layout()
    plt.savefig("jet_pt_by_particle_type.png")

    # Plotting eta distribution
    plt.figure(figsize=(12, 6))
    plt.hist(cumulative_etas, bins=Nbins, alpha=0.5, label='All c-jets', color='orange', range=(-2.5, 2.5), density=True, histtype='step')
    plt.hist(jet_etas_by_type["Other"], bins=Nbins, alpha=0.7, label='Other jets', color='blue', range=(-2.5, 2.5), density=True, histtype='step')
    plt.xlabel('Jet Eta')
    plt.ylabel('Count')
    plt.title('Distribution of Jet Eta by Particle Type')
    plt.legend()
    plt.tight_layout()
    plt.savefig("jet_eta_by_particle_type.png")
