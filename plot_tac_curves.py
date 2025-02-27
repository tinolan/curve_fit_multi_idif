import os
import glob
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from utils.utils_torch import torch_interp_1d, torch_conv
from utils.set_root_paths import root_data_path, root_idif_path
import matplotlib.patches as mpatches


IDIF_SOURCES = {
    "liver": "portal_vein_split_IDIF_ownSeg",
    "lung": "pulmonary_artery_IDIF_ownSeg",
    "kidney": "ureter_IDIF_ownSeg",
    "bladder": "ureter_IDIF_ownSeg"
}

def reduce_to_600(values):
    step = len(values) // 600
    reduced_values = [max(values[i:i+step]) for i in range(0, len(values), step)]
    return torch.tensor(reduced_values[:600], dtype=torch.float32)


class KineticModel_2TC_curve_fit():
    def __init__(self, patient, organ):
        self.patient = patient
        self.organ = organ
    
    def read_idif(self, sample_time, t):
        idif_type = IDIF_SOURCES[self.organ]
        if self.organ == "liver":
            aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{self.patient}_IDIF.txt")
            data = pd.read_csv(aorta_idif_txt_path, sep="\t")
            self.aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.aorta_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.aorta_idif))

            portal_idif_txt_path = os.path.join(root_idif_path, "portal_vein_split_IDIF_ownSeg", f"IDIF_Patient_{self.patient}.txt")
            data = pd.read_csv(portal_idif_txt_path, sep="\t")
            self.portal_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.portal_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.portal_idif))

            return self.aorta_idif_interp, self.portal_idif_interp
        
        elif self.organ == "lung":
            aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{self.patient}_IDIF.txt")
            data = pd.read_csv(aorta_idif_txt_path, sep="\t")
            self.aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.aorta_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.aorta_idif))

            pulmonary_idif = os.path.join(root_idif_path, "pulmonary_artery_IDIF_ownSeg", f"IDIF_Patient_{self.patient}.txt")
            data = pd.read_csv(pulmonary_idif, sep="\t")
            self.pulmonary_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.pulmonary_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.pulmonary_idif))

            return self.aorta_idif_interp, self.pulmonary_idif_interp
        
        elif self.organ == "kidney":
            aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{self.patient}_IDIF.txt")
            data = pd.read_csv(aorta_idif_txt_path, sep="\t")
            self.aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.aorta_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.aorta_idif))

            ureter_idif_txt_path = os.path.join(root_idif_path, "ureter_IDIF_ownSeg", f"IDIF_Patient_{self.patient}.txt")
            data = pd.read_csv(ureter_idif_txt_path, sep="\t")
            self.ureter_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.ureter_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.ureter_idif))

            return self.aorta_idif_interp, self.ureter_idif_interp
        
        elif self.organ == "bladder":
            ureter_idif_txt_path = os.path.join(root_idif_path, "ureter_IDIF_ownSeg", f"IDIF_Patient_{self.patient}.txt")
            data = pd.read_csv(ureter_idif_txt_path, sep="\t")
            self.ureter_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.ureter_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.ureter_idif))

            aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{self.patient}_IDIF.txt")
            data = pd.read_csv(aorta_idif_txt_path, sep="\t")
            self.aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
            self.aorta_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.aorta_idif))

            return self.aorta_idif_interp, self.ureter_idif_interp
        
        else:
            raise ValueError(f"Organ {self.organ} not supported.")
        
        
    def PET_2TC_KM(self, t, k1, k2, k3, Vb, alpha=None, beta=None):
        k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])

        step = 0.1
        if self.organ == "bladder":
            a = self.ureter_idif_interp
        elif self.organ == "liver":
            a = alpha * self.aorta_idif_interp + beta * self.portal_idif_interp
        elif self.organ == "lung":
            a = alpha * self.aorta_idif_interp + beta * self.pulmonary_idif_interp 
        elif self.organ == "kidney":
            a = alpha * self.aorta_idif_interp
        else:
            raise ValueError(f"Organ {self.organ} not supported.")
        
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1 - Vb) * c + Vb * a
        if self.organ == "kidney":
            PET = (1 - Vb) * c + Vb * (a + beta * self.ureter_idif_interp)


        return PET

    def PET_normal(self, t, k1, k2, k3, Vb):
        k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])

        step = 0.1
        a = self.aorta_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1 - Vb) * c + Vb * a

        return PET

def process_patient_organ(patient, organ, organ_path_vb, organ_path_normal, organ_path):
    print(f"Processing {organ} for Patient {patient}")

    time_stamp = torch.load(os.path.join(root_data_path, "DynamicPET/time_stamp.pt"))
    t = torch.load(os.path.join(root_data_path, "DynamicPET/t.pt"))

    KM_2TC = KineticModel_2TC_curve_fit(patient, organ)
    aorta_idif, organ_idif = KM_2TC.read_idif(time_stamp, t)

    param_file_vb = os.path.join(organ_path_vb, f"{patient}_params.npz")
    param_file_normal = os.path.join(organ_path_normal, f"{patient}_params.npz")
    
    if not os.path.exists(param_file_vb) or not os.path.exists(param_file_normal):
        print(f"Missing parameter files for {organ}, {patient}")
        return
    
    param_data_vb = np.load(param_file_vb)
    param_data_normal = np.load(param_file_normal)

    label_map_path = glob.glob(os.path.join(root_data_path, f"DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/labels.nii.gz"))[0]
    label_map = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
    if organ != "bladder":
        label_map = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
        if organ == "liver":
            organ_mask = (label_map == 3) 
        elif organ == "lung":
            organ_mask = np.isin(label_map[:, :, :], [4, 5])
        elif organ == "kidney": 
            organ_mask = np.isin(label_map[:, :, :], [6, 7])
        else:
            raise ValueError(f"Organ {organ} not supported.")
    else: 
        label_map_path = glob.glob(f"/home/guests/valentin_langer/data/segmentationsAndResample/*DynamicFDG_{patient}/urinary_bladder.nii.gz")[0]
        label_map = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
        organ_mask = (label_map[:, :, :] == 1) 

    PET_list = glob.glob(os.path.join(root_data_path, f"DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/PET_*.nii.gz"))
    pet_images = [sitk.GetArrayFromImage(sitk.ReadImage(pet_path)) for pet_path in PET_list]
    
    organ_coords = np.argwhere(organ_mask)
    if organ == "bladder":
        print(organ_coords)
    tac_matrix = []
    for coord in organ_coords:
        x, y, z = coord
        tac_values = [pet_image[x, y, z] / 1000 for pet_image in pet_images]
        tac_tensor = torch.tensor(tac_values, dtype=torch.float32)
        tac_interp = reduce_to_600(torch_interp_1d(t, time_stamp, tac_tensor))
        tac_matrix.append(tac_interp.numpy())
    
    tac_matrix = np.array(tac_matrix)
    tac_mean = np.mean(tac_matrix, axis=0)
    std_per_timepoint = np.std(tac_matrix, axis=0)

    k1_values = param_data_vb['k1'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]
    k2_values = param_data_vb['k2'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]
    k3_values = param_data_vb['k3'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]
    Vb_values = param_data_vb['Vb'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]    
    k1_mean, k2_mean, k3_mean, Vb_mean = np.mean(k1_values), np.mean(k2_values), np.mean(k3_values), np.mean(Vb_values)

    alpha_values = param_data_vb['alpha'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]] if 'alpha' in param_data_vb else None
    beta_values = param_data_vb['beta'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]] if 'beta' in param_data_vb else None
    alpha_mean = np.mean(alpha_values) if alpha_values is not None else None
    beta_mean = np.mean(beta_values) if beta_values is not None else None
    print(k1_mean, k2_mean, k3_mean, Vb_mean, alpha_mean, beta_mean)

    predicted_tac_vb = KM_2TC.PET_2TC_KM(t, k1_mean, k2_mean, k3_mean, Vb_mean, alpha_mean, beta_mean).numpy()

    k1_values = param_data_normal['k1'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]
    k2_values = param_data_normal['k2'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]
    k3_values = param_data_normal['k3'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]
    Vb_values = param_data_normal['Vb'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]]    
    k1_mean, k2_mean, k3_mean, Vb_mean = np.mean(k1_values), np.mean(k2_values), np.mean(k3_values), np.mean(Vb_values)   
    predicted_tac_normal = KM_2TC.PET_normal(t, k1_mean, k2_mean, k3_mean, Vb_mean).numpy()

    predicted_tacs_normal = np.array([
    KM_2TC.PET_normal(t, k1, k2, k3, Vb).numpy()
    for k1, k2, k3, Vb in zip(
        param_data_normal['k1'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_normal['k2'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_normal['k3'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_normal['Vb'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        )
    ])

    predicted_tacs_Vb = np.array([
    KM_2TC.PET_2TC_KM(t, k1, k2, k3, Vb, alpha, beta).numpy()
    for k1, k2, k3, Vb, alpha, beta in zip(
        param_data_vb['k1'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_vb['k2'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_vb['k3'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_vb['Vb'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]],
        param_data_vb['alpha'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]] if 'alpha' in param_data_vb else [None] * len(organ_coords),
        param_data_vb['beta'][organ_coords[:, 0], organ_coords[:, 1], organ_coords[:, 2]] if 'beta' in param_data_vb else [None] * len(organ_coords)
        )
    ])

    predicted_tac_std_normal = np.std(predicted_tacs_normal, axis=0)
    predicted_tac_std_Vb = np.std(predicted_tacs_Vb, axis=0)

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    reduced_std_Vb = np.clip(predicted_tac_std_Vb[np.clip(np.searchsorted(t, time_stamp, side="right") - 1, 0, len(predicted_tac_vb) - 1)], 0, None)
    reduced_std_normal = np.clip(predicted_tac_std_normal[np.clip(np.searchsorted(t, time_stamp, side="right") - 1, 0, len(predicted_tac_normal) - 1)], 0, None)

    # First subplot: Full 60-minute TAC
    axes[0].plot(t.numpy(), tac_mean, label='Real TAC', color='blue')
    axes[0].plot(t.numpy(), predicted_tac_normal, label='Predicted TAC single input', color='red', linestyle='dashed')
    axes[0].plot(t.numpy(), predicted_tac_vb, label='Predicted TAC dual input', color='purple', linestyle='dashed')
    axes[0].fill_between(t.numpy(), np.clip(tac_mean - std_per_timepoint, 0, None), np.clip(tac_mean + std_per_timepoint, 0, None), color='blue', alpha=0.2)
    axes[0].errorbar(
        time_stamp.numpy()[35:],
        predicted_tac_normal[np.clip(np.searchsorted(t, time_stamp, side="right") - 1, 0, len(predicted_tac_normal) - 1)[35:]],
        yerr=reduced_std_normal[35:],
        fmt="o",
        color="red",
        elinewidth=0.5,
        markersize=3,  
        label="Predicted TAC single input (error)"
    )
    axes[0].errorbar(
        time_stamp.numpy()[35:],
        predicted_tac_std_Vb[np.clip(np.searchsorted(t, time_stamp, side="right") - 1, 0, len(predicted_tac_std_Vb) - 1)[35:]],
        yerr=reduced_std_Vb[35:],
        fmt="o",
        color="purple",
        elinewidth=0.5,
        markersize=3,  
        label="Predicted TAC dual input (error)"
    )
    axes[0].set_title("TAC over 60 Minutes")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("kBq/ml")
    handles, labels = axes[1].get_legend_handles_labels()
    fill_patch = mpatches.Patch(color='blue', alpha=0.2, label="Real TAC ± STD")
    handles.append(fill_patch)
    axes[0].legend(handles=handles, loc="upper right")
    axes[0].grid(True)

    unique_indices, unique_inverse = np.unique(np.searchsorted(t.numpy(), time_stamp.numpy()[:38], side="right"), return_index=True)

    reduced_std_Vb = np.clip(predicted_tac_std_Vb[unique_indices], 0, None)
    reduced_std_normal = np.clip(predicted_tac_std_normal[unique_indices], 0, None)
    time_point = 31

    aorta_idif_scaled = aorta_idif * (50 / aorta_idif.max())

    axes[1].plot(t.numpy()[:time_point], tac_mean[:time_point], label='Real TAC', color='blue')
    axes[1].plot(t.numpy()[:time_point], predicted_tac_normal[:time_point], label='Predicted TAC single input', color='red', linestyle='dashed')
    axes[1].plot(t.numpy()[:time_point], predicted_tac_vb[:time_point], label='Predicted TAC dual input', color='purple', linestyle='dashed')
    if organ == "liver":
        axes[1].plot(t.numpy()[:time_point], aorta_idif_scaled[:time_point], label='Aorta IDIF', color='green', linestyle='dotted')
        axes[1].plot(t.numpy()[:time_point], organ_idif[:time_point], label='Portal Vein IDIF', color='orange', linestyle='dotted')
    elif organ == "lung":
        organ_idif_scaled = organ_idif * (50 / organ_idif.max())
        axes[1].plot(t.numpy()[:time_point], aorta_idif_scaled[:time_point], label='Aorta IDIF', color='green', linestyle='dotted')
        axes[1].plot(t.numpy()[:time_point], organ_idif_scaled[:time_point], label='Pulmonary Artery IDIF', color='orange', linestyle='dotted')
    elif organ == "kidney":
        axes[1].plot(t.numpy()[:time_point], aorta_idif_scaled[:time_point], label='Aorta IDIF', color='green', linestyle='dotted')
        axes[1].plot(t.numpy()[:time_point], organ_idif[:time_point], label='Ureter IDIF', color='orange', linestyle='dotted')
    elif organ == "bladder":
        axes[1].plot(t.numpy()[:time_point], organ_idif[:time_point], label='Ureter IDIF', color='green', linestyle='dotted')
        axes[1].plot(t.numpy()[:time_point], aorta_idif_scaled[:time_point], label='Aorta IDIF', color='orange', linestyle='dotted')
    axes[1].fill_between(t.numpy()[:time_point], np.clip(tac_mean[:time_point] - std_per_timepoint[:time_point], 0, None), np.clip(tac_mean[:time_point] + std_per_timepoint[:time_point], 0, None), color='blue', alpha=0.2)
    axes[1].errorbar(
        t.numpy()[unique_indices],
        predicted_tac_normal[unique_indices],
        yerr=reduced_std_normal,
        fmt="o",
        color="red",
        elinewidth=0.5,
        markersize=3,  # Smaller markers
        label="Predicted TAC single input (error)"
    )
    axes[1].errorbar(
        t.numpy()[unique_indices],
        predicted_tac_vb[unique_indices],
        yerr=reduced_std_Vb,
        fmt="o",
        color="purple",
        elinewidth=0.5,
        markersize=3,  # Smaller markers
        label="Predicted TAC multi input (error)"
    )
    ax2 = axes[1].twinx()
    ax2.set_ylim(0, aorta_idif.max())
    ax2.set_ylabel("Aorta IDIF (kBq/ml)")
    axes[1].set_title("TAC in first 3 Minutes")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("kBq/ml")
    handles, labels = axes[1].get_legend_handles_labels()
    fill_patch = mpatches.Patch(color='blue', alpha=0.2, label="Real TAC ± STD")
    handles.append(fill_patch)
    axes[1].legend(handles=handles, loc="upper right")
    axes[1].grid(True)

    fig.savefig(os.path.join(organ_path, f"{patient}_mean_tacovertime_with_errors.png"))



def process_all_patients():
    base_path = "/home/guests/valentin_langer/data/plots/organ_curve/image-derived"
    organs = ["liver", "lung", "kidney", "bladder"]
    
    for organ in organs:
        organ_dirs = [d for d in os.listdir(base_path) if d.startswith(organ) and os.path.isdir(os.path.join(base_path, d))]
        
        for organ_dir in organ_dirs:
            if "Vb" not in organ_dir and "normal" not in organ_dir:
                continue 
            if "06" not in organ_dir:
                continue
            print("starting", organ_dir)
            parts = organ_dir.split("_")
            patient = parts[-1]
            
            if "Vb" in organ_dir:
                organ_path_vb = os.path.join(base_path, organ_dir)
                organ_path_normal = os.path.join(base_path, f"{organ}_normal_{patient}")
                if os.path.exists(organ_path_normal):
                    save_path = os.path.join(base_path, organ_dir)
                    process_patient_organ(patient, organ, organ_path_vb, organ_path_normal, save_path)



if __name__ == "__main__":
    "start plotting MSE"
    process_all_patients()