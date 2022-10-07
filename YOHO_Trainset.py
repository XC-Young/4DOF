"""
Generate Trainset using 3dmatch_train for PartI and PartII.
for pair in pairs:
    random rotation --> pc0 pc1
    random z rotation --> pc1
    feature extraction --> feats0 feats1
    point pairs selection (already generated)
    save -- batch
"""

import os
import numpy as np
import argparse
import open3d as o3d
import torch
import random
from tqdm import tqdm
from utils.r_eval import compute_R_diff,quaternion_from_matrix
from utils.dataset import get_dataset_name
from utils.utils import make_non_exists_dir,random_rotation_matrix,random_z_rotation,read_pickle,save_pickle
from utils.misc import extract_features
from fcgf_model import load_model


class trainset_create():
    def __init__(self,setname='3dmatch_train'):
        self.dataset_name=setname
        self.origin_data_dir='./data/origin_data'
        self.datasets=get_dataset_name(self.dataset_name,self.origin_data_dir)
        self.output_dir='./data/YOHO_FCGF'
        self.Rgroup=np.load('./group_related/Rotation_8.npy')
        self.valscenes=self.datasets['valscenes']
        self.batchsize=32
        self.val_num=5000

    def PCA_keys_sample(self):
        for name,dataset in tqdm(self.datasets.items()):
            if name in ['wholesetname','valscenes']:continue

            Save_keys_dir=f'{self.output_dir}/Filtered_Keys/{dataset.name}'
            Save_pair_dir=f'{self.output_dir}/Pairs_0.03/{dataset.name}'
            make_non_exists_dir(Save_keys_dir)
            make_non_exists_dir(Save_pair_dir)

            for pc_id in tqdm(dataset.pc_ids): #index in pc
                if os.path.exists(f'{Save_keys_dir}/{pc_id}_index.npy'):continue
                Keys_index=np.loadtxt(dataset.get_key_dir(pc_id)).astype(np.int)
                Keys=dataset.get_kps(pc_id)
                # Pcas=np.load(f'{dataset.root}/pca_0.3/{pc_id}.npy')
                Pcas=np.ones_like(Keys)
                Ok_index=np.arange(Pcas.shape[0])[Pcas[:,0]>0.03].astype(np.int)
                Keys=Keys[Ok_index]
                Keys_index=Keys_index[Ok_index]
                #Save the filtered index
                np.save(f'{Save_keys_dir}/{pc_id}_coor.npy',Keys)
                np.save(f'{Save_keys_dir}/{pc_id}_index.npy',Keys_index) #in pc
            
            #pair with the filtered keypoints: index in keys
            for pair in tqdm(dataset.pair_ids):
                pc0,pc1=pair
                if os.path.exists(f'{Save_pair_dir}/{pc0}-{pc1}.npy'):continue
                keys0=torch.from_numpy(np.load(f'{Save_keys_dir}/{pc0}_coor.npy').astype(np.float32)).cuda()
                keys1=torch.from_numpy(np.load(f'{Save_keys_dir}/{pc1}_coor.npy').astype(np.float32)).cuda()
                diff=torch.norm(keys0[:,None,:]-keys1[None,:,:],dim=-1).cpu().numpy()
                pair=np.where(diff<0.02)
                pair=np.concatenate([pair[0][:,None],pair[1][:,None]],axis=1)# pairnum*2
                np.save(f'{Save_pair_dir}/{pc0}-{pc1}.npy',pair)

    
    def FCGF_Group_Feature_Extractor(self,args,Point,Keys_index): #index in pc
        #output:kn*32*8
        output=[]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(args.model)
        config = checkpoint['config']

        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
            
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        model = model.to(device)
        
        for i in range(self.Rgroup.shape[0]):
            one_R_output=[]
            R_i=self.Rgroup[i]
            Point_i=Point@R_i.T
            Keys_i=Point_i[Keys_index]
            with torch.no_grad():
                xyz_down, feature = extract_features(
                                    model,
                                    xyz=Point_i,
                                    voxel_size=config.voxel_size,
                                    device=device,
                                    skip_check=True)
            feature=feature.cpu().numpy()
            xyz_down_pcd = o3d.geometry.PointCloud()
            xyz_down_pcd.points = o3d.utility.Vector3dVector(xyz_down)
            pcd_tree = o3d.geometry.KDTreeFlann(xyz_down_pcd)
            for k in range(Keys_i.shape[0]):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(Keys_i[k], 1)
                one_R_output.append(feature[idx[0]][None,:])
            one_R_output=np.concatenate(one_R_output,axis=0)#kn*32
            output.append(one_R_output[:,:,None])
        return np.concatenate(output,axis=-1) #kn*32*8


    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id


    def DeltaR(self,R,index):
        R_anchor=self.Rgroup[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)


    # Trainset generating
    def trainset(self,args,repeat_dataset=3):
        batch_id=0
        Save_list_dir=f'{self.output_dir}/Train_val_list/trainset'
        make_non_exists_dir(Save_list_dir)
        for Repeat in range(repeat_dataset):
            for key,dataset in tqdm(self.datasets.items()):
                if key in ['wholesetname','valscenes']:continue
                for pair in tqdm(dataset.pair_ids):
                    id0,id1=pair
                    #random rotation, R_id, R_residule
                    pc0=dataset.get_pc(id0)                    
                    pc1=dataset.get_pc(id1)         
                    R_base=random_rotation_matrix()
                    pc0=pc0@R_base.T
                    pc1=pc1@R_base.T
                    R_z=random_z_rotation(180)
                    pc1=pc1@R_z.T   
                    R_index=self.R2DR_id(R_z)
                    R_residule=self.DeltaR(R_z,R_index)
                    R_zs,R_indexs,R_residules=[],[],[]
                    for b in range(self.batchsize):
                        R_zs.append(R_z[None,:,:])
                        R_indexs.append(R_index)
                        R_residules.append(R_residule[None,:])
                    R_zs=np.concatenate(R_zs,axis=0)
                    R_indexs=np.array(R_indexs)
                    R_residules=np.concatenate(R_residules,axis=0)                      
                    #pc0_feat
                    Key_idx0=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id0}_index.npy')
                    feats0=self.FCGF_Group_Feature_Extractor(args,pc0,Key_idx0)
                    #pc1_feat
                    Key_idx1=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id1}_index.npy')
                    feats1=self.FCGF_Group_Feature_Extractor(args,pc1,Key_idx1)                   
                    #sample pps
                    Key_pps=np.load(f'{self.output_dir}/Pairs_0.03/{dataset.name}/{id0}-{id1}.npy') #index in keys
                    pps_all=np.arange(Key_pps.shape[0]) #index
                    if pps_all.shape[0]<10:continue
                    if pps_all.shape[0]<self.batchsize:
                        pps_all=np.repeat(pps_all,int(self.batchsize/pps_all.shape[0])+1)
                        np.random.shuffle(pps_all)
                    np.random.shuffle(pps_all)
                    pps=Key_pps[pps_all[0:self.batchsize]]# bn*2
                    #sample keys
                    keys0=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id0}_coor.npy')  
                    keys_sample0=keys0[pps[:,0]]
                    keys1=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id1}_coor.npy')  
                    keys1=dataset.get_kps(id1)  
                    keys_sample1=keys1[pps[:,1]] 
                    #sample feats
                    feats0=feats0[pps[:,0]]
                    feats1=feats1[pps[:,1]]
                    #save                   
                    item={
                        'feats0':torch.from_numpy(feats0.astype(np.float32)), #before enhanced rot
                        'feats1':torch.from_numpy(feats1.astype(np.float32)), #after enhanced rot
                        'keys0':torch.from_numpy(keys_sample0.astype(np.float32)),
                        'keys1':torch.from_numpy(keys_sample1.astype(np.float32)),
                        'R':torch.from_numpy(R_zs.astype(np.float32)),
                        'true_idx':torch.from_numpy(R_indexs.astype(np.int)),
                        'deltaR':torch.from_numpy(R_residules.astype(np.float32))
                    }                  
                    torch.save(item,f'{Save_list_dir}/{batch_id}.pth',_use_new_zipfile_serialization=False)
                    batch_id+=1
        save_pickle(range(batch_id),f'{self.output_dir}/Train_val_list/train.pkl')
                

    def valset(self):
        Save_list_dir=f'{self.output_dir}/Train_val_list/valset'
        make_non_exists_dir(Save_list_dir)
        batch_id=0
        val_list=[]
        for scene in tqdm(self.valscenes):
            dataset=self.datasets[scene]
            for pair in tqdm(dataset.pair_ids):
                id0,id1=pair
                val_list.append((scene,id0,id1))
        random.shuffle(val_list)
        for pair in val_list[0:self.val_num]:
            scene,id0,id1=pair
            dataset=self.datasets[scene]
            #random rotation, R_id, R_residule
            pc0=dataset.get_pc(id0)                    
            pc1=dataset.get_pc(id1)         
            R_base=random_rotation_matrix()
            pc0=pc0@R_base.T
            pc1=pc1@R_base.T
            R_z=random_z_rotation(180)
            pc1=pc1@R_z.T   
            R_index=self.R2DR_id(R_z)
            R_residule=self.DeltaR(R_z,R_index)
            R_zs,R_indexs,R_residules=[],[],[]
            for b in range(self.batchsize):
                R_zs.append(R_z[None,:,:])
                R_indexs.append(R_index)
                R_residules.append(R_residule[None,:])
            R_zs=np.concatenate(R_zs,axis=0)
            R_indexs=np.array(R_indexs)
            R_residules=np.concatenate(R_residules,axis=0)                      
            #pc0_feat
            Key_idx0=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id0}_index.npy')
            feats0=self.FCGF_Group_Feature_Extractor(args,pc0,Key_idx0)
            #pc1_feat
            Key_idx1=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id1}_index.npy')
            feats1=self.FCGF_Group_Feature_Extractor(args,pc1,Key_idx1)                   
            #sample pps
            Key_pps=np.load(f'{self.output_dir}/Pairs_0.03/{dataset.name}/{id0}-{id1}.npy') #index in keys
            pps_all=np.arange(Key_pps.shape[0]) #index
            if pps_all.shape[0]<10:continue
            if pps_all.shape[0]<self.batchsize:
                pps_all=np.repeat(pps_all,int(self.batchsize/pps_all.shape[0])+1)
                np.random.shuffle(pps_all)
            np.random.shuffle(pps_all)
            pps=Key_pps[pps_all[0:self.batchsize]]# bn*2
            #sample keys
            keys0=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id0}_coor.npy')  
            keys_sample0=keys0[pps[:,0]]
            keys1=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{id1}_coor.npy')  
            keys_sample1=keys1[pps[:,1]] 
            #sample feats
            feats0=feats0[pps[:,0]]
            feats1=feats1[pps[:,1]]
            #save                   
            item={
                'feats0':torch.from_numpy(feats0.astype(np.float32)), #before enhanced rot
                'feats1':torch.from_numpy(feats1.astype(np.float32)), #after enhanced rot
                'keys0':torch.from_numpy(keys_sample0.astype(np.float32)),
                'keys1':torch.from_numpy(keys_sample1.astype(np.float32)),
                'R':torch.from_numpy(R_zs.astype(np.float32)),
                'true_idx':torch.from_numpy(R_indexs.astype(np.int)),
                'deltaR':torch.from_numpy(R_residules.astype(np.float32))
            }                  
            torch.save(item,f'{Save_list_dir}/{batch_id}.pth',_use_new_zipfile_serialization=False)
            batch_id+=1
        save_pickle(range(batch_id),f'{self.output_dir}/Train_val_list/val.pkl')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        default='./model/Backbone/best_val_checkpoint.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--datasetname',
        default='3dmatch_train',
        type=str,
        help='trainset name')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    args = parser.parse_args()

    trainset_creater=trainset_create(setname=args.datasetname)
    trainset_creater.PCA_keys_sample()
    trainset_creater.trainset(args)
    trainset_creater.valset()
    