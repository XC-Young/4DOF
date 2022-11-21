from email import generator
import os
from re import sub
import torch
import numpy as np
import glob
import open3d as o3d
from tqdm import tqdm
from utils.utils import *
from fcgf_model import load_model
from utils.misc import extract_features
from utils.r_eval import compute_R_diff

'''
pc0: scan PC
pc1: stereo PC
Ground truth transformation is applied to pc1
'''

class scst_generate:
  def __init__(self):
    self.testseq = [8,9]
    self.scstdir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/kittiscst'
    self.predir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/YOHO_FCGF/Testset'
    self.load_model()
    self.G = np.load(f'./group_related/Rotation_8.npy')

  def apply_transform(self, pts, trans):
      R = trans[:3, :3]
      T = trans[:3, 3]
      pts = pts @ R.T + T
      return pts

  def loadset(self):
    self.scst = {}
    for i in range(8,10):
      seq = {
            'pc':[],
            'pair':{}
            }
      fn = f'{self.scstdir}/{i}/PointCloud/gt.log'
      with open(fn,'r') as f:
        lines = f.readlines()
        pair_num = len(lines)//5
        for k in range(pair_num):
          id0,id1=np.fromstring(lines[k*5],dtype=np.float32,sep='\t')[0:2]
          id0=int(id0)
          id1=int(id1)
          row0=np.fromstring(lines[k*5+1],dtype=np.float32,sep=' ')
          row1=np.fromstring(lines[k*5+2],dtype=np.float32,sep=' ')
          row2=np.fromstring(lines[k*5+3],dtype=np.float32,sep=' ')
          row3=np.fromstring(lines[k*5+4],dtype=np.float32,sep=' ')
          transform=np.stack([row0,row1,row2,row3])
          seq['pair'][f'{id0}-{id1}'] = transform
          if not id0 in seq['pc']:
              seq['pc'].append(id0)
          if not id1 in seq['pc']:
              seq['pc'].append(id1)
      self.scst[f'{i}'] = seq

  # generate key points for stereo and scan pc
  def generate_kps(self):
    for i in self.testseq:
        seq = self.scst[f'{i}']
        kpsdir = f'{self.scstdir}/{i}/Keypoints'
        make_non_exists_dir(kpsdir)
        kpspcdir = f'{self.scstdir}/{i}/Keypoints_PC'
        make_non_exists_dir(kpspcdir)
        for pc in tqdm(seq['pc']):
            pc = int(pc)
            kpsfn = f'{kpsdir}/cloud_bin_{pc}Keypoints.txt'
            pcd = np.load(f'{self.scstdir}/{i}/PointCloud/cloud_bin_{pc}.npy')
            index = np.arange(pcd.shape[0])
            np.random.shuffle(index)
            index = index[0:5000]
            np.savetxt(kpsfn, index)
            kpcd = pcd[index]
            np.save(f'{kpspcdir}/cloud_bin_{pc}Keypoints.npy',kpcd)

  def load_model(self):
      checkpoint = torch.load('./model/Backbone/best_val_checkpoint.pth')
      config = checkpoint['config']
      Model = load_model(config.model)
      num_feats = 1
      self.model = Model(
          num_feats,
          config.model_n_out,
          bn_momentum=0.05,
          normalize_feature=config.normalize_feature,
          conv1_kernel_size=config.conv1_kernel_size,
          D=3)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model.to(self.device)
      self.model.load_state_dict(checkpoint['state_dict'])
      self.model.eval()

  def generate_scan_gfeats(self,pc,key):
      feats = []
      if pc.shape[0]>40000:
          index = np.arange(pc.shape[0])
          np.random.shuffle(index)
          pc = pc[index[0:40000]]
      for gid in range(self.G.shape[0]):
          feats_g = []
          g = self.G[gid]
          #rot the point cloud
          pc_g = pc@g.T
          key_g = key@g.T
          with torch.no_grad():
              pc_g_down, feature_g = extract_features(
                                  self.model,
                                  xyz=pc_g,
                                  voxel_size=0.3,
                                  device=self.device,
                                  skip_check=True)
          feature_g=feature_g.cpu().numpy()
          xyz_down_pcd = o3d.geometry.PointCloud()
          xyz_down_pcd.points = o3d.utility.Vector3dVector(pc_g_down)
          pcd_tree = o3d.geometry.KDTreeFlann(xyz_down_pcd)
          for k in range(key_g.shape[0]):
              [_, idx, _] = pcd_tree.search_knn_vector_3d(key_g[k], 1)
              feats_g.append(feature_g[idx[0]][None,:])
          feats_g=np.concatenate(feats_g,axis=0)#kn*32
          feats.append(feats_g[:,:,None])
      feats = np.concatenate(feats, axis=-1)#kn*32*8
      return feats

  def generate_test_gfeats(self):
      for i in self.testseq:
          seq = self.scst[f'{i}']
          savedir = f'./data/YOHO_FCGF/Testset/kittiscst/{i}/FCGF_Input_Group_feature'
          make_non_exists_dir(savedir)
          for pc in tqdm(seq['pc']):
            feats = []
            # load pointcloud and keypoints
            xyz = np.load(f'{self.scstdir}/{i}/PointCloud/cloud_bin_{pc}.npy')
            # key = np.loadtxt(f'{self.scstdir}/{i}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
            # key = xyz[key]
            key = np.load(f'{self.scstdir}/{i}/Keypoints_PC/cloud_bin_{pc}Keypoints.npy')
            feats = self.generate_scan_gfeats(xyz, key)
            np.save(f'{savedir}/{pc}.npy', feats)

if __name__=='__main__':
  generator = scst_generate()
  generator.loadset()
#   generator.generate_kps()
  generator.generate_test_gfeats()