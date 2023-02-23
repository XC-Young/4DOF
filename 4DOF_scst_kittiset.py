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

  def sample_random_trans_z(self, pcd):
    T = np.eye(4)
    rng = np.random.RandomState()
    theta = np.pi * rng.uniform(-1.0, 1.0)
    alpha = theta
    Rzalpha=np.array([[np.cos(alpha),np.sin(alpha),0],
                      [-np.sin(alpha),np.cos(alpha),0],
                      [0,0,1]])
    R = Rzalpha
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

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

  # to add random transformation to pc1
  def load_save_pc(self):
    for i in self.testseq:
      seq = self.scst[f'{i}']
      srcdir = f'{self.scstdir}/{i}/PointCloud'
      if i==8:
        savedir = f'{self.scstdir}/8_rotation'
        make_non_exists_dir(savedir)
        rotdir = f'{savedir}/rotation'
        make_non_exists_dir(rotdir)
        for pc in tqdm(seq['pc']):
          pc = int(pc)
          if pc<1000:
            pc1 = np.load(f'{srcdir}/cloud_bin_{pc}.npy')
            np.save(f'{savedir}/cloud_bin_{pc}.npy',pc1)
            ply1 = o3d.geometry.PointCloud()
            ply1.points = o3d.utility.Vector3dVector(pc1)
            o3d.io.write_point_cloud(f'{savedir}/cloud_bin_{pc}.ply',ply1)
          else:
            pc0 = np.load(f'{srcdir}/cloud_bin_{pc}.npy')
            # add random transformation to pc0
            T_z = self.sample_random_trans_z(pc0)
            pc0 = self.apply_transform(pc0,T_z)
            np.save(f'{rotdir}/{pc}.npy',T_z)
            np.save(f'{savedir}/cloud_bin_{pc}.npy',pc0)
            ply0 = o3d.geometry.PointCloud()
            ply0.points = o3d.utility.Vector3dVector(pc0)
            o3d.io.write_point_cloud(f'{savedir}/cloud_bin_{pc}.ply',ply0)
      if i==9:
        savedir = f'{self.scstdir}/9_rotation'
        make_non_exists_dir(savedir)
        rotdir = f'{savedir}/rotation'
        make_non_exists_dir(rotdir)
        for pc in tqdm(seq['pc']):
          pc = int(pc)
          if pc>1000:
            pc1 = np.load(f'{srcdir}/cloud_bin_{pc}.npy')
            np.save(f'{savedir}/cloud_bin_{pc}.npy',pc1)
            ply1 = o3d.geometry.PointCloud()
            ply1.points = o3d.utility.Vector3dVector(pc1)
            o3d.io.write_point_cloud(f'{savedir}/cloud_bin_{pc}.ply',ply1)
          else:
            pc0 = np.load(f'{srcdir}/cloud_bin_{pc}.npy')
            # add random transformation to pc0
            T_z = self.sample_random_trans_z(pc0)
            pc0 = self.apply_transform(pc0,T_z)
            np.save(f'{rotdir}/{pc}.npy',T_z)
            np.save(f'{savedir}/cloud_bin_{pc}.npy',pc0)
            ply0 = o3d.geometry.PointCloud()
            ply0.points = o3d.utility.Vector3dVector(pc0)
            o3d.io.write_point_cloud(f'{savedir}/cloud_bin_{pc}.ply',ply0)

      gtfn = f'{savedir}/gt.log'
      writer=open(gtfn,'w')
      pc_num = len(seq['pc'])
      gt = np.eye(4)
      for pair,trans in tqdm(seq['pair'].items()):
        id0,id1 = str.split(pair,'-')
        T_z = np.load(f'{rotdir}/{id0}.npy')
        # gt = trans @ np.linalg.inv(T_z)
        gt = T_z @ trans
        transform_pr = gt
        writer.write(f'{int(id0)}\t{int(id1)}\t{pc_num}\n')
        writer.write(f'{transform_pr[0,0]}\t{transform_pr[0,1]}\t{transform_pr[0,2]}\t{transform_pr[0,3]}\n')
        writer.write(f'{transform_pr[1,0]}\t{transform_pr[1,1]}\t{transform_pr[1,2]}\t{transform_pr[1,3]}\n')
        writer.write(f'{transform_pr[2,0]}\t{transform_pr[2,1]}\t{transform_pr[2,2]}\t{transform_pr[2,3]}\n')
        writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
    writer.close()  

  # generate key points for stereo and scan pc
  def generate_kps(self):
    for i in self.testseq:
        seq = self.scst[f'{i}']
        kpsdir = f'{self.scstdir}/{i}/Keypoints'
        make_non_exists_dir(kpsdir)
        kpspcdir = f'{self.scstdir}/{i}/Keypoints_PC'
        make_non_exists_dir(kpspcdir)
        dispdir = f'{self.scstdir}/{i}/Disparity'
        make_non_exists_dir(dispdir)
        for pair,trans in tqdm(seq['pair'].items()):
          id0,id1 = str.split(pair,'-')
          # lidar
          kpsfn0 = f'{kpsdir}/cloud_bin_{id0}Keypoints.txt'
          pcd0 = np.load(f'{self.scstdir}/{i}/PointCloud/cloud_bin_{id0}.npy')
          index0 = np.arange(pcd0.shape[0])
          np.random.shuffle(index0)
          index0 = index0[0:5000]
          np.savetxt(kpsfn0, index0)
          kpcd0 = pcd0[index0]
          np.save(f'{kpspcdir}/cloud_bin_{id0}Keypoints.npy',kpcd0)
          # stereo
          kpsfn1 = f'{kpsdir}/cloud_bin_{id1}Keypoints.txt'
          disp = np.load(f'{self.scstdir}/{i}/PointCloud/disparity/disp_{id1}.npy')
          xyz1 = np.load(f'{self.scstdir}/{i}/PointCloud/cloud_bin_{id1}.npy')
          # to connect disparity and xyz
          ply1 = o3d.geometry.PointCloud()
          ply1.points = o3d.utility.Vector3dVector(xyz1)
          color = np.c_[disp,disp,disp]
          ply1.colors = o3d.utility.Vector3dVector(color)
          # o3d.io.write_point_cloud(f'{kpspcdir}/cloud_bin_{id1}.ply',ply1)
          # voxel dowmsample
          ply1 = ply1.voxel_down_sample(0.35)
          pcd1 = np.array(ply1.points).astype(np.float32)
          disps = np.array(ply1.colors).astype(np.float32)[:,0]
          lenth = pcd1.shape[0]
          assert lenth>=5000
          index1 = np.arange(pcd1.shape[0])
          np.random.shuffle(index1)
          index1 = index1[0:5000]
          np.savetxt(kpsfn1, index1)
          # save keypoints' disparity
          dispss = disps[index1]
          # d_max = np.max(dispss)
          # d_min = np.min(dispss)
          # dispss = (dispss-d_min)/(d_max-d_min)
          np.save(f'{dispdir}/disp_{id1}.npy',dispss)
          # save keypoints' pc
          kpcd1 = pcd1[index1]
          np.save(f'{kpspcdir}/cloud_bin_{id1}Keypoints.npy',kpcd1)
          # save xyz&disp
          kpsxyzddir = f'{self.scstdir}/{i}/Keypoints_xyzd'
          make_non_exists_dir(kpsxyzddir)
          xyzd = np.concatenate([kpcd1,dispss[:,None]],axis=-1)
          np.savetxt(f'{kpsxyzddir}/cloud_bin_{id1}.txt',xyzd)


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
  generator.generate_kps()
  # generator.load_save_pc()
  generator.generate_test_gfeats()