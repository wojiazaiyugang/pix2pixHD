import os.path
from pathlib import Path
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = make_dataset(self.dir_A)

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = make_dataset(self.dir_B)

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = make_dataset(self.dir_inst)

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = make_dataset(self.dir_feat)

        self.dataset_size = len(self.A_paths) 
    
    def _get_concatenated_frames(self, index):
        B_tensors = []
        for i in range(-2, 3):
            current_index = max(13, min(index + i, len(self.A_paths) - 1))
            B_path = self.B_paths[current_index]   
            B = Image.open(B_path).convert('RGB')
            params = get_params(self.opt, B.size)
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
            B_tensors.append(B_tensor)

        return torch.cat(B_tensors, dim=0)
    

    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        audio_data = np.load(str(Path(A_path).with_suffix(".npy")))
        # audio_data = audio_data.reshape(16, 32, 32)
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = C_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

            C_tensor = self._get_concatenated_frames(index)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))


        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'imagelist': C_tensor,
                      'feat': feat_tensor, 'path': A_path, "audio": audio_data}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'