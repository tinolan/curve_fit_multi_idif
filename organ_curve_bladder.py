import os
import glob
from natsort import natsorted
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.utils_torch import torch_interp_1d, torch_conv
from utils.set_root_paths import root_data_path, root_idif_path

def reduce_to_600(values):
    step = len(values) // 600  
    reduced_values = [max(values[i:i+step]) for i in range(0, len(values), step)]
    return torch.tensor(reduced_values[:600], dtype=torch.float32)

class KineticModel_2TC_curve_fit():
    def __init__(self, patient):
        self.patient = patient
    
    def read_idif(self, sample_time, t):
        aorta_idif_txt_path = os.path.join(root_data_path, "DynamicPET/IDIF", f"DynamicFDG_{self.patient}_IDIF.txt")
        data = pd.read_csv(aorta_idif_txt_path, sep="\t")
        self.aorta_idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.aorta_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.aorta_idif))
    
        ureter_idif_txt_path = os.path.join(root_idif_path, "ureter_IDIF_ownSeg", f"IDIF_Patient_{self.patient}.txt")
        data = pd.read_csv(ureter_idif_txt_path, sep="\t")
        self.ureter_idif = torch.Tensor(data["plasma[kBq/cc]"])
        self.ureter_idif_interp = reduce_to_600(torch_interp_1d(t, sample_time, self.ureter_idif))

        return self.ureter_idif_interp
    
    def PET_2TC_KM(self, t, k1, k2, k3, Vb,):
        k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])
        step = 0.1
        a = self.ureter_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1-Vb) * c + Vb * a
        return PET

    def PET_normal(self, t, k1, k2, k3, Vb,):
        k1, k2, k3, Vb = map(lambda x: torch.tensor(x, dtype=torch.float32), [k1, k2, k3, Vb])
        step = 0.1
        a = self.aorta_idif_interp
        e = (k2 + k3) * t
        b = k1 / (k2 + k3) * (k3 + k2 * torch.exp(-e))
        c = torch_conv(a, b) * step
        PET = (1-Vb) * c + Vb * a
        return PET

def process_pet_bladder(patient):
    print(f"Starting ureter bladder processing for patient {patient}")
    time_stamp = torch.load(root_data_path + "/DynamicPET/time_stamp.pt")
    t_file = torch.load(root_data_path + "/DynamicPET/t.pt")
    t = torch.linspace(0, t_file[-1], 2000)

    label_map_path = glob.glob(f"/home/guests/valentin_langer/data/segmentationsAndResample/*DynamicFDG_{patient}/urinary_bladder.nii.gz")[0]
    PET_list = natsorted(glob.glob(root_data_path + f"/DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/PET_*.nii.gz"))
    save_path = os.path.join("/home/guests/valentin_langer/data/plots/organ_curve", "image-derived", f"bladder_Vbt_{patient}")
    os.makedirs(save_path, exist_ok=True)
    
    label_map_ = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
    bladder_mask = (label_map_[:, :, :] == 1) 
    
    pet_images = [sitk.GetArrayFromImage(sitk.ReadImage(pet_path)) for pet_path in PET_list]
    
    KM_2TC = KineticModel_2TC_curve_fit(patient)
    ureter_idif = KM_2TC.read_idif(time_stamp, t_file)
    
    bladder_coords = np.argwhere(bladder_mask)
    tac_matrix = []
    predicted_tac_matrix = []
    param_maps = {param: torch.full(label_map_.shape, 0, dtype=torch.float32) for param in ["k1", "k2", "k3", "Vb"]}

    #bladder_coords = bladder_coords[:5]  
    
    for coord in bladder_coords:
        z, x, y = coord
        tac_values = [pet_image[z, x, y] / 1000 for pet_image in pet_images]
        tac_tensor = torch.tensor(tac_values, dtype=torch.float32)

        auc = torch.trapezoid(tac_tensor, time_stamp)
        if auc < 10:
            continue

        tac_interp = reduce_to_600(torch_interp_1d(t, time_stamp, tac_tensor))

        tac_matrix.append(tac_interp)

        try:
            p, _ = curve_fit(KM_2TC.PET_2TC_KM, t_file, tac_interp,
                             p0=[0.1, 0.1, 0.001, 0.001],
                             bounds=([0.01, 0.01, 0.001, 0.001], [10, 10, 1, 0.05]),
                             diff_step=0.001)
            k1, k2, k3, Vb = p

            param_maps["k1"][z, x, y] = k1
            param_maps["k2"][z, x, y] = k2
            param_maps["k3"][z, x, y] = k3
            param_maps["Vb"][z, x, y] = Vb

            predicted_tac = KM_2TC.PET_2TC_KM(t_file, k1, k2, k3, Vb).detach().numpy()
            predicted_tac_matrix.append(predicted_tac)

        except RuntimeError:
            print(f"Fehler für Voxel ({z}, {x}, {y})")
            continue
    param_save_path = os.path.join(save_path, f"{patient}_params.npz")
    np.savez_compressed(param_save_path, **param_maps)


def process_pet_bladder_normal(patient):
    print(f"Starting normal bladder processing for patient {patient}")
    time_stamp = torch.load(root_data_path + "/DynamicPET/time_stamp.pt")
    t_file = torch.load(root_data_path + "/DynamicPET/t.pt")
    t = torch.linspace(0, t_file[-1], 2000)

    label_map_path = glob.glob(f"/home/guests/valentin_langer/data/segmentationsAndResample/*DynamicFDG_{patient}/urinary_bladder.nii.gz")[0]
    PET_list = natsorted(glob.glob(root_data_path + f"/DynamicPET/*DynamicFDG_{patient}/NIFTY/Resampled/PET_*.nii.gz"))
    save_path = os.path.join("/home/guests/valentin_langer/data/plots/organ_curve", "image-derived", f"bladder_normalt_{patient}")
    os.makedirs(save_path, exist_ok=True)
    
    label_map_ = sitk.GetArrayFromImage(sitk.ReadImage(label_map_path))
    bladder_mask = (label_map_[:, :, :] == 1) 
    
    pet_images = [sitk.GetArrayFromImage(sitk.ReadImage(pet_path)) for pet_path in PET_list]
    
    KM_2TC = KineticModel_2TC_curve_fit(patient)
    ureter_idif = KM_2TC.read_idif(time_stamp, t_file)
    
    bladder_coords = np.argwhere(bladder_mask)
    tac_matrix = []
    predicted_tac_matrix = []
    param_maps = {param: torch.full(label_map_.shape, 0, dtype=torch.float32) for param in ["k1", "k2", "k3", "Vb"]}

    #bladder_coords = bladder_coords[:5] 
    
    for coord in bladder_coords:
        z, x, y = coord
        tac_values = [pet_image[z, x, y] / 1000 for pet_image in pet_images]
        tac_tensor = torch.tensor(tac_values, dtype=torch.float32)

        auc = torch.trapezoid(tac_tensor, time_stamp)
        if auc < 10:
            continue
        tac_interp = reduce_to_600(torch_interp_1d(t, time_stamp, tac_tensor))

        tac_matrix.append(tac_interp)

        try:
            p, _ = curve_fit(KM_2TC.PET_normal, t_file, tac_interp,
                             p0=[0.1, 0.1, 0.001, 0.001],
                             bounds=([0.01, 0.01, 0.001, 0.001], [10, 10, 1, 0.05]),
                             diff_step=0.001)
            k1, k2, k3, Vb = p

            param_maps["k1"][z, x, y] = k1
            param_maps["k2"][z, x, y] = k2
            param_maps["k3"][z, x, y] = k3
            param_maps["Vb"][z, x, y] = Vb

            predicted_tac = KM_2TC.PET_normal(t_file, k1, k2, k3, Vb).detach().numpy()
            predicted_tac_matrix.append(predicted_tac)

        except RuntimeError:
            print(f"Fehler für Voxel ({z}, {x}, {y})")
            continue
    param_save_path = os.path.join(save_path, f"{patient}_params.npz")
    np.savez_compressed(param_save_path, **param_maps)

    
if __name__ == '__main__':
    process_pet_bladder(patient="10")
    process_pet_bladder_normal(patient="10")