import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
import os

class SafeFastRNADataset(Dataset):

    def __init__(self, 
                 feature_npy_path: str, 
                 label_npy_path: str,
                 meta_csv_path: str, 
                 task: str,
                 target_dim: int = 1729,
                 dss_matrix = None,       
                 doid_map: dict = None,  
                 doid_col_name: str = 'DO_ID' 
                 ): 
        
        self.task = task
        self.target_dim = target_dim
        

        self.features = np.load(feature_npy_path, mmap_mode='r')
        

        self.labels = np.load(label_npy_path, allow_pickle=True)
        
     
        self.dss_matrix = dss_matrix
        self.doid_map = doid_map
        self.meta_indices = None
        
        if self.task == 'association':
            if meta_csv_path is None or not os.path.exists(meta_csv_path):
                import warnings
                warnings.warn(f"Meta CSV not found ({meta_csv_path}); DSS features will not be appended.")
            else:
             
                df_meta = pd.read_csv(meta_csv_path)
                
                if doid_col_name not in df_meta.columns:
                    raise ValueError(f"Meta CSV 缺少列: {doid_col_name}")
                
              
                if self.dss_matrix is not None and self.doid_map is not None:
                     self.meta_indices = [self.doid_map.get(d, 0) for d in df_meta[doid_col_name]]
                     self.meta_indices = np.array(self.meta_indices, dtype=np.int32)
                     
                     final_dim = self.features.shape[1] + self.dss_matrix.shape[1]
                     if final_dim != self.target_dim:
                         self.target_dim = final_dim
        
        if len(self.features) != len(self.labels):
             raise ValueError(f"特征数({len(self.features)})与标签数({len(self.labels)})不匹配")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].copy() 
        
        if self.task == 'association' and self.dss_matrix is not None and self.meta_indices is not None:
            dss_row_idx = self.meta_indices[idx]
            dss_feat = self.dss_matrix[dss_row_idx] # (128,)
            feat = np.concatenate([feat, dss_feat])
        
    
        current_dim = feat.shape[0]
        if current_dim < self.target_dim:
            pad_len = self.target_dim - current_dim
            pads = np.zeros(pad_len, dtype=feat.dtype)
            feat = np.concatenate([feat, pads])
        
      
        base_feat = torch.from_numpy(feat).float()
        
 
        raw_label = self.labels[idx]
        
        if self.task == 'family':
            return {
                'features': base_feat,
                'family_label': torch.tensor(raw_label, dtype=torch.long)
            }
            
        elif self.task == 'association':
            return {
                'features': base_feat,       
                'association_label': torch.tensor(raw_label, dtype=torch.float32)
            }