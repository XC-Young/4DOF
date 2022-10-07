from email import generator
import os
from re import sub
import torch
import numpy as np
import glob, random
import open3d as o3d
from tqdm import tqdm
from utils.utils import *
from fcgf_model import load_model
from utils.misc import extract_features
from utils.r_eval import compute_R_diff

class subseq_generate:
  def __init__(self):
    self.testseq = [8,9,10]
    self.stereodir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/KittiStereo'
    self.scandir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/kitti'
    self.savedir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/subseq'
    self.predir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/YOHO_FCGF/Testset'
    self.load_model()
    self.downsamplevalue = 0.2
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

  def stereo(self):
    fns = sorted(glob.glob(f'{self.stereodir}/d>=8/10/PointCloud/cloud_bin_*.ply'))
    prefns = glob.glob(f'{self.predir}/KittiStereo/1110_d>8_1.5_2/10/Match/YOHO_C/1000iters/*.npz')
    ids = []
    for fn in fns:
      id = str.split(fn,'/')[-1]
      id = str.split(str.split(id,'_')[-1],'.')[0]
      ids.append(int(id))
    ids = sorted(ids)
    for i in range(len(fns)):
      fns[i] = f'{self.stereodir}/d>=8/10/PointCloud/cloud_bin_'+str(ids[i])+'.ply'
    preids1 = []
    preids2 = []
    for prefn in prefns:
      preid = str.split(prefn,'/')[-1]
      preid = str.split(preid,'.')[0]
      preid1 = str.split(preid,'-')[0]
      preid2 = str.split(preid,'-')[1]
      preids1.append(int(preid1))
      preids2.append(int(preid2))
    preids1 = sorted(preids1)
    preids2 = sorted(preids2)
    for i in range(len(prefns)):
      prefns[i] = f'{self.predir}/KittiStereo/1110_d>8_1.5_2/10/Match/YOHO_C/1000iters/'+str(preids1[i])+'-'+str(preids2[i])+'.npz'

    for j in tqdm(range(int((len(fns)-4)/8))):
      pcds = {
      'id':[],
      'pc':[],
      'trans':[]
      }
      for i in range(8):
        # id = ids[i+4+j*8]
        id = ids[i+1+j*8]
        pcds['id'].append(id)
        # pcd = o3d.io.read_point_cloud(fns[i+4+j*8])
        pcd = o3d.io.read_point_cloud(fns[i+1+j*8])
        pcd_d = o3d.geometry.PointCloud.voxel_down_sample(pcd, self.downsamplevalue)
        pcd_d = np.array(pcd_d.points)
        pcds['pc'].append(pcd_d)
        # trans = np.load(prefns[i+3+j*8])['trans']
        trans = np.load(prefns[i+j*8])['trans']
        pcds['trans'].append(trans)
      pcds['pc'][7] = self.apply_transform(pcds['pc'][7] ,pcds['trans'][7])
      pcds['pc'][6] = np.concatenate([pcds['pc'][6],pcds['pc'][7]],axis=0)
      pcds['pc'][6] = self.apply_transform(pcds['pc'][6] ,pcds['trans'][6])
      pcds['pc'][5] = np.concatenate([pcds['pc'][5],pcds['pc'][6]],axis=0)
      pcds['pc'][5] = self.apply_transform(pcds['pc'][5] ,pcds['trans'][5])
      pcds['pc'][4] = np.concatenate([pcds['pc'][4],pcds['pc'][5]],axis=0)
      pcds['pc'][4] = self.apply_transform(pcds['pc'][4] ,pcds['trans'][4])
      pcds['pc'][3] = np.concatenate([pcds['pc'][3],pcds['pc'][4]],axis=0)
      pcds['pc'][3] = self.apply_transform(pcds['pc'][3] ,pcds['trans'][3])
      pcds['pc'][2] = np.concatenate([pcds['pc'][2],pcds['pc'][3]],axis=0)
      pcds['pc'][2] = self.apply_transform(pcds['pc'][2] ,pcds['trans'][2])
      pcds['pc'][1] = np.concatenate([pcds['pc'][1],pcds['pc'][2]],axis=0)
      pcds['pc'][1] = self.apply_transform(pcds['pc'][1] ,pcds['trans'][1])
      pcds['pc'][0] = np.concatenate([pcds['pc'][0],pcds['pc'][1]],axis=0)
      ply = o3d.geometry.PointCloud()
      ply.points = o3d.utility.Vector3dVector(pcds['pc'][0])
      pc1 = str(pcds['id'][0])
      pc2 = str(pcds['id'][7])
      o3d.io.write_point_cloud(f'{self.savedir}/stereo/10/cloud_bin_{pc1}_{pc2}.ply',ply)

  def stereo_gt(self):
    fns = sorted(glob.glob(f'{self.stereodir}/d>=8/10/PointCloud/cloud_bin_*.ply'))
    gtfns = glob.glob(f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/KITTI/icp/10_*')
    ids = []
    for fn in fns:
      id = str.split(fn,'/')[-1]
      id = str.split(str.split(id,'_')[-1],'.')[0]
      ids.append(int(id))
    ids = sorted(ids)
    for i in range(len(fns)):
      fns[i] = f'{self.stereodir}/d>=8/10/PointCloud/cloud_bin_'+str(ids[i])+'.ply'

    gtids1 = []
    gtids2 = []
    for gtfn in gtfns:
      gtid = str.split(gtfn,'/')[-1][:-4]
      gtid1 = str.split(gtid,'_')[1]
      gtid2 = str.split(gtid,'_')[2]
      gtids1.append(int(gtid1))
      gtids2.append(int(gtid2))
    gtids1 = sorted(gtids1)
    gtids2 = sorted(gtids2)
    for i in range(len(gtfns)):
      gtfns[i] = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/KITTI/icp/10_'+str(gtids1[i])+'_'+str(gtids2[i])+'.npy'

    for j in tqdm(range(int((len(fns)-4)/8))):
      pcds = {
      'id':[],
      'pc':[],
      'trans':[]
      }
      for i in range(8):
        id = ids[i+1+j*8]
        # id = ids[i+4+j*8]
        pcds['id'].append(id)
        pcd = o3d.io.read_point_cloud(fns[i+1+j*8])
        # pcd = o3d.io.read_point_cloud(fns[i+4+j*8])
        pcd_d = o3d.geometry.PointCloud.voxel_down_sample(pcd, self.downsamplevalue)
        pcd_d = np.array(pcd_d.points)
        pcds['pc'].append(pcd_d)
        trans = np.linalg.inv(np.load(gtfns[i+j*8]))
        # trans = np.linalg.inv(np.load(gtfns[i+3+j*8]))
        pcds['trans'].append(trans)
      pcds['pc'][7] = self.apply_transform(pcds['pc'][7] ,pcds['trans'][7])
      pcds['pc'][6] = np.concatenate([pcds['pc'][6],pcds['pc'][7]],axis=0)
      pcds['pc'][6] = self.apply_transform(pcds['pc'][6] ,pcds['trans'][6])
      pcds['pc'][5] = np.concatenate([pcds['pc'][5],pcds['pc'][6]],axis=0)
      pcds['pc'][5] = self.apply_transform(pcds['pc'][5] ,pcds['trans'][5])
      pcds['pc'][4] = np.concatenate([pcds['pc'][4],pcds['pc'][5]],axis=0)
      pcds['pc'][4] = self.apply_transform(pcds['pc'][4] ,pcds['trans'][4])
      pcds['pc'][3] = np.concatenate([pcds['pc'][3],pcds['pc'][4]],axis=0)
      pcds['pc'][3] = self.apply_transform(pcds['pc'][3] ,pcds['trans'][3])
      pcds['pc'][2] = np.concatenate([pcds['pc'][2],pcds['pc'][3]],axis=0)
      pcds['pc'][2] = self.apply_transform(pcds['pc'][2] ,pcds['trans'][2])
      pcds['pc'][1] = np.concatenate([pcds['pc'][1],pcds['pc'][2]],axis=0)
      pcds['pc'][1] = self.apply_transform(pcds['pc'][1] ,pcds['trans'][1])
      pcds['pc'][0] = np.concatenate([pcds['pc'][0],pcds['pc'][1]],axis=0)
      ply = o3d.geometry.PointCloud()
      ply.points = o3d.utility.Vector3dVector(pcds['pc'][0])
      pc1 = str(pcds['id'][0])
      pc2 = str(pcds['id'][7])
      o3d.io.write_point_cloud(f'{self.savedir}/stereo_gt/10/cloud_bin_{pc1}_{pc2}.ply',ply)

  def scan(self):
    fns = sorted(glob.glob(f'{self.scandir}/10/PointCloud/cloud_bin_*.ply'))
    prefns = glob.glob(f'{self.predir}/kitti/10/Match/YOHO_C/1000iters/*.npz')
    ids = []
    for fn in fns:
      id = str.split(fn,'/')[-1]
      id = str.split(str.split(id,'_')[-1],'.')[0]
      ids.append(int(id))
    ids = sorted(ids)
    for i in range(len(fns)):
      fns[i] = f'{self.scandir}/10/PointCloud/cloud_bin_'+str(ids[i])+'.ply'
    preids1 = []
    preids2 = []
    for prefn in prefns:
      preid = str.split(prefn,'/')[-1]
      preid = str.split(preid,'.')[0]
      preid1 = str.split(preid,'-')[0]
      preid2 = str.split(preid,'-')[1]
      preids1.append(int(preid1))
      preids2.append(int(preid2))
    preids1 = sorted(preids1)
    preids2 = sorted(preids2)
    for i in range(len(prefns)):
      prefns[i] = f'{self.predir}/kitti/10/Match/YOHO_C/1000iters/'+str(preids1[i])+'-'+str(preids2[i])+'.npz'
    
    for j in tqdm(range(int((len(fns)-4)/8))):
      pcds = {
        'id':[],
        'pc':[],
        'trans':[]
      }
      for i in range(8):
        id = ids[i+1+j*8]
        # id = ids[i+4+j*8]
        pcds['id'].append(id)
        pcd = o3d.io.read_point_cloud(fns[i+1+j*8])
        # pcd = o3d.io.read_point_cloud(fns[i+4+j*8])
        pcd_d = o3d.geometry.PointCloud.voxel_down_sample(pcd, self.downsamplevalue)
        pcd_d = np.array(pcd_d.points)
        pcds['pc'].append(pcd_d)
        trans = np.load(prefns[i+j*8])['trans']
        # trans = np.load(prefns[i+3+j*8])['trans']
        pcds['trans'].append(trans)
      pcds['pc'][7] = self.apply_transform(pcds['pc'][7] ,pcds['trans'][7])
      pcds['pc'][6] = np.concatenate([pcds['pc'][6],pcds['pc'][7]],axis=0)
      pcds['pc'][6] = self.apply_transform(pcds['pc'][6] ,pcds['trans'][6])
      pcds['pc'][5] = np.concatenate([pcds['pc'][5],pcds['pc'][6]],axis=0)
      pcds['pc'][5] = self.apply_transform(pcds['pc'][5] ,pcds['trans'][5])
      pcds['pc'][4] = np.concatenate([pcds['pc'][4],pcds['pc'][5]],axis=0)
      pcds['pc'][4] = self.apply_transform(pcds['pc'][4] ,pcds['trans'][4])
      pcds['pc'][3] = np.concatenate([pcds['pc'][3],pcds['pc'][4]],axis=0)
      pcds['pc'][3] = self.apply_transform(pcds['pc'][3] ,pcds['trans'][3])
      pcds['pc'][2] = np.concatenate([pcds['pc'][2],pcds['pc'][3]],axis=0)
      pcds['pc'][2] = self.apply_transform(pcds['pc'][2] ,pcds['trans'][2])
      pcds['pc'][1] = np.concatenate([pcds['pc'][1],pcds['pc'][2]],axis=0)
      pcds['pc'][1] = self.apply_transform(pcds['pc'][1] ,pcds['trans'][1])
      pcds['pc'][0] = np.concatenate([pcds['pc'][0],pcds['pc'][1]],axis=0)

      ply = o3d.geometry.PointCloud()
      ply.points = o3d.utility.Vector3dVector(pcds['pc'][0])
      pc1 = str(pcds['id'][0])
      pc2 = str(pcds['id'][7])
      o3d.io.write_point_cloud(f'{self.savedir}/scan/10/cloud_bin_{pc1}_{pc2}.ply',ply)

  def loadset_forICP(self):
    self.subseq = {}
    for i in self.testseq:
      # gt
      fn = f'{self.savedir}/gt.log'
      with open(f'/home/hdmap/yxcdata/03_Data/KITTI/02_stereo/calib/{i}.txt','r') as f:
        calib = f.readlines()
        Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
        rotationX = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
        rotationZ = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
        gt = Tr_velo_to_cam*rotationZ*rotationX

      subseq = {
        'pc':[],
        'pair':{}
      }
      stfns = glob.glob(f'{self.savedir}/stereo/{i}/could_bin_*.ply')
      scfns = glob.glob(f'{self.savedir}/scan/{i}/could_bin_*.ply')
      for fn in stfns:
        pair = str.split(fn,'/')[-1]
        pair = str.split(pair,'.')[0]
        pair = str.split(pair,'_')[-2:]
        subseq['pair'][f'{pair[0]}-{pair[1]}'] = np.array(gt)
        if not pair[0] in subseq['pc']:
          subseq['pc'].append(pair[0])
        if not pair[1] in subseq['pc']:
          subseq['pc'].append(pair[1])
        assert len(subseq['pair'].keys())*2 == len(subseq['pc'])
      self.subseq[f'{i}'] = subseq
    
  def icp(self,init = np.eye(4)):
    for i in self.testseq:
      seq = self.subseq[f'{i}']
      icpdir = f'{self.savedir}/icp'
      make_non_exists_dir(icpdir)
      stdir = f'{self.savedir}/stereo/{i}'
      scdir = f'{self.savedir}/scan/{i}'
      for pair,trans in tqdm(seq['pair'].items()):
        id0,id1 = str.split(pair,'-')
        icpfn = f'{icpdir}/{i}_{id0}_{id1}.npy'
        ply0 = o3d.io.read_point_cloud(f'{stdir}/could_bin_{id0}_{id1}.ply')
        ply1 = o3d.io.read_point_cloud(f'{scdir}/could_bin_{id0}_{id1}.ply')
        reg = o3d.registration.registration_icp(
              ply0, ply1, 0.2, init,
              o3d.registration.TransformationEstimationPointToPoint(),
              o3d.registration.ICPConvergenceCriteria(max_iteration=200))
        icp = reg.transformation
        tdiff=np.sum(np.square(icp[0:3,-1]-trans[0:3,-1]))
        Rdiff=compute_R_diff(trans[0:3,0:3],icp[0:3,0:3])
        np.save(icpfn,icp)
        if tdiff>1 or Rdiff>5:
          print(id0,id1)

  def loadset(self):
    self.subseq = {}
    for i in self.testseq:
        seq = {
            'pc':[],
            'pair':{}
            }
        pair_fns = glob.glob(f'{self.savedir}/icp/{i}_*')
        for fn in pair_fns:
            trans = np.load(fn)
            pair = str.split(fn,'/')[-1][:-4]
            pair = str.split(pair,'_')
            assert int(pair[0]) == i
            seq['pair'][f'{pair[1]}-{pair[2]}'] = trans
            if not pair[1] in seq['pc']:
                seq['pc'].append(pair[1])
            if not pair[2] in seq['pc']:
                seq['pc'].append(pair[2])
            assert len(seq['pair'].keys())*2 == len(seq['pc'])
        self.subseq[f'{i}'] = seq

  def load_save_pc(self):
    for i in self.testseq:
      seq = self.subseq[f'{i}']
      stdir = f'{self.savedir}/stereo/{i}'
      scdir = f'{self.savedir}/scan/{i}'
      savedir = f'{self.savedir}/{i}/PointCloud'
      make_non_exists_dir(savedir)
      gtfn = f'{self.savedir}/{i}/PointCloud/gt.log'
      writer=open(gtfn,'w')
      pc_num = len(seq['pc'])
      for pair,trans in tqdm(seq['pair'].items()):
        id0,id1 = str.split(pair,'-')
        ply0 = o3d.io.read_point_cloud(f'{stdir}/could_bin_{id0}_{id1}.ply')
        ply1 = o3d.io.read_point_cloud(f'{scdir}/could_bin_{id0}_{id1}.ply')
        pc0 = np.array(ply0.points)
        pc1 = np.array(ply1.points)
        # add random transformation to pc1
        T_z = self.sample_random_trans_z(pc1)
        pc1 = self.apply_transform(pc1,T_z)
        gt = np.linalg.inv(trans@T_z)
        np.save(f'{savedir}/cloud_bin_{id0}.npy',pc0)
        np.save(f'{savedir}/cloud_bin_{id1}.npy',pc1)
        ply1 = o3d.geometry.PointCloud()
        ply1.points = o3d.utility.Vector3dVector(pc1)
        o3d.io.write_point_cloud(f'{savedir}/cloud_bin_{id0}.ply',ply0)
        o3d.io.write_point_cloud(f'{savedir}/cloud_bin_{id1}.ply',ply1)
        
        transform_pr = gt
        writer.write(f'{int(id0)}\t{int(id1)}\t{pc_num}\n')
        writer.write(f'{transform_pr[0,0]}\t{transform_pr[0,1]}\t{transform_pr[0,2]}\t{transform_pr[0,3]}\n')
        writer.write(f'{transform_pr[1,0]}\t{transform_pr[1,1]}\t{transform_pr[1,2]}\t{transform_pr[1,3]}\n')
        writer.write(f'{transform_pr[2,0]}\t{transform_pr[2,1]}\t{transform_pr[2,2]}\t{transform_pr[2,3]}\n')
        writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
    writer.close()  

  def generate_kps(self):
    for i in self.testseq:
      seq = self.subseq[f'{i}']
      kpsdir = f'{self.savedir}/{i}/Keypoints'
      make_non_exists_dir(kpsdir)
      for p in tqdm(seq['pc']):
        p = int(p)
        kpsfn = f'{kpsdir}/cloud_bin_{p}Keypoints.txt'
        pcd = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{p}.npy')
        index = np.arange(pcd.shape[0])
        np.random.shuffle(index)
        index = index[0:5000]
        np.savetxt(kpsfn,index)

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
          seq = self.subseq[f'{i}']
          savedir = f'./data/YOHO_FCGF/Testset/subseq/{i}/FCGF_Input_Group_feature'
          make_non_exists_dir(savedir)
          for pc in tqdm(seq['pc']):
              feats = []
              # load pointcloud and keypoints
              xyz = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.npy')
              key = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
              key = xyz[key]
              feats = self.generate_scan_gfeats(xyz, key)
              np.save(f'{savedir}/{pc}.npy', feats)


if __name__=='__main__':
  generator = subseq_generate()
  # generator.stereo()
  # generator.stereo_gt()
  # generator.scan()
  # generator.loadset_forICP()
  # generator.icp()
  generator.loadset()
  # generator.load_save_pc()
  generator.generate_kps()
  generator.generate_test_gfeats()

