import torch
from torch.utils.data import Dataset
import uproot
import numpy as np

class ParticleJetDataset(Dataset):
    def __init__(self, root_files, reduce_ds=0, Nfeatures = 10, evaluation = False, npad = None):
        
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
            #self.full_data_array["part_energy"][idx],
            #self.full_data_array["part_massReco"][idx],
            # target label -> Remove before training !!!
            #self.full_data_array["part_isFromD"][idx]
        ], axis=1)

        part_features = part_features[:, :self.Nfeatures]  # Select only the first Nfeatures

        # **Convert one-hot labels to three-class indices**
        isFromD = self.full_data_array["part_isFromD"][idx]
        isFromDStar = self.full_data_array["part_isFromDStar"][idx]

        # Convert to class index:
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

        return (torch.tensor(part_features, dtype=torch.float32),
                torch.tensor(labels_particle, dtype=torch.long),
                torch.tensor(label_jet, dtype=torch.float32))
    
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