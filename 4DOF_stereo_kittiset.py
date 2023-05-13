import os
import torch
import cv2
import numpy as np
import glob, random
from utils.utils import *
from utils.r_eval import *
import open3d as o3d
from tqdm import tqdm
import multiprocessing as mp
from fcgf_model import load_model
from knn_search import knn_module
from utils.misc import extract_features

class generate_kitti:
    def __init__(self):
        self.trainseq = [0,1,2,3,4,5]
        self.valseq = [6,7]
        self.testseq = [8,9,10]
        self.basedir = f'/home/hdmap/yxcdata/03_Data/KITTI/01_odometry'
        self.savedir = f'./data/origin_data/kittistereo'
        make_non_exists_dir(self.savedir)
        self.yohosavedir = f'./data/YOHO_FCGF/kittistereo'
        self.load_model()
        self.G = np.load(f'./group_related/Rotation_8.npy')
        self.knn=knn_module.KNN(1)
        self.batchsize = 64
        self.n_train_rs = 2

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts
    
    def loadset(self):
        self.kitti = {}
        for i in range(11):
            seq = {
                'pc':[],
                'pair':{}
                }
            pair_fns = glob.glob(f'{self.basedir}/icp/{i}_*')
            # pair_fns = glob.glob(f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/KITTI/icp/{i}_*')
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
            self.kitti[f'{i}'] = seq

    def Disp2PC(self):
        print('Disparity to point cloud')
        for i in self.valseq:
            seq = self.kitti[f'{i}']
            seqname = f'{i}'.zfill(2)
            pc_save = f'{self.savedir}/{i}/PointCloud'
            make_non_exists_dir(pc_save)
            disp_save = f'{pc_save}/disparity'
            make_non_exists_dir(disp_save)
            calibfile = open(f'{self.basedir}/data_odometry_calib/dataset/sequences/{seqname}/calib.txt','r')
            calibdata = calibfile.readlines()
            for row in calibdata:
                caliblist = row.split(' ')
                if caliblist[0]=='P2:':
                    fu1 = float(caliblist[1])
                    cu1 = float(caliblist[3])
                    fubs1 = float(caliblist[4])
                    fv1 = float(caliblist[6])
                    cv1 = float(caliblist[7])
                if caliblist[0]=='P3:':
                    fu2 = float(caliblist[1])
                    fubs2 = float(caliblist[4])
            baseline = abs(-fubs2/fu2+fubs1/fu1) #0.5379044881339211
            for pair,trans in tqdm(seq['pair'].items()):
                pc0,pc1 = str.split(pair,'-')
                imgname = f'{pc1}'.zfill(6)
                colorimg = cv2.imread(f'{self.basedir}/data_odometry_color/dataset/sequences/{seqname}/image_2/{imgname}.png')
                depthimg = cv2.imread(f'{self.basedir}/data_odometry_color/dataset/sequences/{seqname}/Disp/{imgname}.png',cv2.IMREAD_UNCHANGED)
                # Crop the size of color image to depth image directly
                colorimg = cv2.resize(colorimg,(1232,368),interpolation=cv2.INTER_LINEAR)
                points = []
                rgbs = []
                disps = []
                for row in range(depthimg.shape[0]):
                    for col in range(depthimg.shape[1]):
                        d = depthimg[row, col]
                        d = d/255.0
                        if d>=8:
                            z_origin = fu1*baseline/d
                            x_origin = (col-cu1)*z_origin/fu1
                            y_origin = (row-cv1)*z_origin/fv1
                            # first rotate -90 across X, then rotate -90 across Z
                            x = z_origin
                            y = -x_origin
                            z = -y_origin
                            bgr = colorimg[row, col]
                            rgb = np.array([bgr[2],bgr[1],bgr[0]])
                            rgbs.append(rgb[None,:])
                            point = np.array([x,y,z])
                            points.append(point[None,:])
                            disps.append(d)
                rgbs = np.concatenate(rgbs,axis=0)/255
                points = np.concatenate(points,axis=0)
                disps = np.array(disps)
                ply = o3d.geometry.PointCloud()
                ply.points = o3d.utility.Vector3dVector(points)
                ply.colors = o3d.utility.Vector3dVector(rgbs)
                o3d.io.write_point_cloud(f'{pc_save}/cloud_bin_{pc1}.ply',ply)
                np.save(f'{pc_save}/cloud_bin_{pc1}.npy',points)
                np.save(f'{disp_save}/disp_{pc1}.npy',disps)

    def Disp2PC_kitti360(self):
      print('Disparity to point cloud')
      pc_save = f'{self.savedir}/PointCloud'
      make_non_exists_dir(pc_save)
      disp_save = f'{pc_save}/disparity'
      make_non_exists_dir(disp_save)
      fu1 = 552.554261
      cu1 = 682.049453-91
      # cu1 = 682.049453
      fv1 = 552.554261
      cv1 = 238.769549-3
      # cv1 = 238.769549
      # baseline = 328.318735/fu1
      baseline = 0.60
      T_cam2vel = np.array([[0.04307104361,-0.08829286498,0.995162929,0.8043914418],[-0.999004371,0.007784614041,0.04392796942,0.2993489574],[-0.01162548558,-0.9960641394,-0.08786966659,-0.1770225824]])
      fns = glob.glob(f'{self.imgdir}/image_00/data_rect/*.png')
      for fn in tqdm(fns):
          name = str.split(fn,'/')[-1]
          name = str.split(name,'.')[0]
          pc_num = int(name)
          colorimg = cv2.imread(f'{self.imgdir}/image_00/data_rect/{name}.png')
          depthimg = cv2.imread(f'{self.imgdir}/Disp_lac/{name}.png',cv2.IMREAD_UNCHANGED)
          colorimg = colorimg[3:373,91:1317]
          colorimg = cv2.resize(colorimg,(1232,368),interpolation=cv2.INTER_LINEAR)
          points = []
          rgbs = []
          disps = []
          for row in range(depthimg.shape[0]):
              for col in range(depthimg.shape[1]):
                  d = depthimg[row, col]
                  d = d/255.0
                  if d>=8:
                      z = fu1*baseline/d
                      x = (col-cu1)*z/fu1
                      y = (row-cv1)*z/fv1
                      bgr = colorimg[row, col]
                      rgb = np.array([bgr[2],bgr[1],bgr[0]])
                      rgbs.append(rgb[None,:])
                      point = np.array([x,y,z])
                      point = self.apply_transform(point,T_cam2vel)
                      points.append(point[None,:])
                      disps.append(d)
          rgbs = np.concatenate(rgbs,axis=0)/255
          points = np.concatenate(points,axis=0)
          disps = np.array(disps)
          ply = o3d.geometry.PointCloud()
          ply.points = o3d.utility.Vector3dVector(points)
          ply.colors = o3d.utility.Vector3dVector(rgbs)
          o3d.io.write_point_cloud(f'{pc_save}/cloud_bin_{pc_num}.ply',ply)
          np.save(f'{disp_save}/disp_{pc_num}.npy',disps)

    def generate_gt(self):
        print('Generate gr.npy')
        for i in self.testseq:
            seq = self.kitti[f'{i}']
            seq_name = f'{i}'.zfill(2)
            with open(f'/home/hdmap/yxcdata/03_Data/KITTI/02_stereo/calib/{seq_name}.txt','r') as f:
                calib = f.readlines()
            # P2 (3 x 4) for left eye
            P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
            R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
            # Add a 1 in bottom-right, reshape to 4 x 4
            R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
            R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
            Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
            Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
            pc2img = P2 * R0_rect * Tr_velo_to_cam
            fu = P2[0,0]
            fv = P2[1,1]
            cu = P2[0,2]
            cv = P2[1,2]
            K = np.array([[fu,0,cu],[0,fv,cv],[0,0,1]])
            K_inv = np.linalg.inv(K)
            gt_seq = K_inv*P2*R0_rect*Tr_velo_to_cam
            gt_seq = np.insert(gt_seq,3,values=[0,0,0,1],axis=0)
            for pair,trans in tqdm(seq['pair'].items()):
                pc0,pc1 = str.split(pair,'-')
                trans = np.linalg.inv(trans)
                rotationX = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
                rotationZ = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
                gt_sameseq = rotationZ*rotationX*gt_seq
                gt = trans*np.linalg.inv(gt_sameseq) #Stero & Scan
                np.save(f'{self.savedir}/GT/{i}_{pc0}_{pc1}.npy',gt)

    def gt_log(self):
        print('Generate gt_log')
        for i in self.testseq:
            seq = self.kitti[f'{i}']
            gt_fns = glob.glob(f'{self.basedir}/icp/{i}_*')
            # gt_fns = glob.glob(f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/KITTI/icp/{i}_*')
            fn = f'{self.savedir}/{i}/PointCloud/gt.log'
            make_non_exists_dir(f'{self.savedir}/{i}/PointCloud')
            writer=open(fn,'w')
            # pc_num = len(seq['pc'])
            pc_num = len(gt_fns)
            for fn in gt_fns:
                transform_pr = np.load(fn)
                pair = str.split(fn,'/')[-1][:-4]
                pair = str.split(pair,'_')
                pc0 = pair[1]
                pc1 = pair[2]
                # it's gt apply to pc1
                transform_pr = np.linalg.inv(transform_pr)
                writer.write(f'{int(pc0)}\t{int(pc1)}\t{pc_num}\n')
                writer.write(f'{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n')
                writer.write(f'{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n')
                writer.write(f'{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n')
                writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
            writer.close()

    # save scan pc.npy
    def load_save_pc(self):
        print('save scan pc.npy')
        for i in self.valseq:
            seq = self.kitti[f'{i}']
            scan_plydir = f'/home/hdmap/yxcdata/02_Codes/YOHO/data/origin_data/kitti/{i}/PointCloud'
            for pair,trans in tqdm(seq['pair'].items()):
                pc0,pc1 = str.split(pair,'-')
                scan_ply = o3d.io.read_point_cloud(f'{scan_plydir}/cloud_bin_{pc0}.ply')
                scan_pcd = np.array(scan_ply.points)
                np.save(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc0}.npy',scan_pcd)

    # generate key points for stereo and scan pc
    def generate_kps(self):
        print('Generate key points for stereo and scan pc.')
        for i in self.testseq:
            seq = self.kitti[f'{i}']
            kpsdir = f'{self.savedir}/{i}/Keypoints'
            make_non_exists_dir(kpsdir)
            dispdir = f'{self.savedir}/{i}/Disparity'
            make_non_exists_dir(dispdir)
            for pair, trans in tqdm(seq['pair'].items()):
                pc0,pc1=str.split(pair,'-')
                pc0 = int(pc0)
                pc1 = int(pc1)
                kpsfn0 = f'{kpsdir}/cloud_bin_{pc0}Keypoints.txt'
                kpsfn1 = f'{kpsdir}/cloud_bin_{pc1}Keypoints.txt'
                # os.system(f'mv {self.savedir}/{key}/PointCloud/{p}.ply {self.savedir}/{key}/PointCloud/cloud_bin_{p}.ply')
                pcd0 = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc0}.npy')
                pcd1 = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc1}.npy')
                index0 = np.arange(pcd0.shape[0])
                index1 = np.arange(pcd1.shape[0])
                np.random.shuffle(index0)
                np.random.shuffle(index1)
                index0 = index0[0:5000]
                index1 = index1[0:5000]
                # save keypoints' disparity
                disp0 = np.load(f'{self.savedir}/{i}/PointCloud/disparity/disp_{pc0}.npy')
                disp1 = np.load(f'{self.savedir}/{i}/PointCloud/disparity/disp_{pc1}.npy')
                disps0 = []
                disps1 = []
                for j in range(5000):
                    disps0.append(disp0[index0[j]])
                    disps1.append(disp1[index1[j]])
                disps0 = np.array(disps0)
                disps1 = np.array(disps1)
                np.save(f'{dispdir}/disp_{pc0}.npy',disp0)
                np.save(f'{dispdir}/disp_{pc1}.npy',disp1)

                np.savetxt(kpsfn0, index0)
                np.savetxt(kpsfn1, index1)
                
    def generate_kps_new(self):
        print('Generate key points for stereo and scan pc.')
        for i in self.valseq:
            seq = self.kitti[f'{i}']
            kpsdir = f'{self.savedir}/{i}/Keypoints'
            make_non_exists_dir(kpsdir)
            kpcddir = f'{self.savedir}/{i}/Keypoints_PC'
            make_non_exists_dir(kpcddir)
            dispdir = f'{self.savedir}/{i}/Disparity'
            make_non_exists_dir(dispdir)
            lenths = []
            for pc in tqdm(seq['pc']):
                kpsfn = f'{kpsdir}/cloud_bin_{pc}Keypoints.txt'
                disp = np.load(f'{self.savedir}/{i}/PointCloud/disparity/disp_{pc}.npy')
                # os.system(f'mv {self.savedir}/{key}/PointCloud/{p}.ply {self.savedir}/{key}/PointCloud/cloud_bin_{p}.ply')
                # pcd = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.npy')

                """ # Disparity normalization
                # inverse weight2
                disp0 = 1/disp0
                disp1 = 1/disp1
                sum0 = np.sum(disp0)
                sum1 = np.sum(disp1)
                disp0=disp0/sum0
                disp1=disp1/sum1
                index0 = np.random.choice( np.arange(pcd0.shape[0]), 5000, p = disp0, replace = False)
                index1 = np.random.choice( np.arange(pcd1.shape[0]), 5000, p = disp1, replace = False) """

                # It's wrong
                """ # voxel down sample
                ply = o3d.io.read_point_cloud(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.ply')
                ply = ply.voxel_down_sample(0.35)
                pcd = np.array(ply.points).astype(np.float32)
                lenth = pcd.shape[0]
                assert lenth>=5000
                index = np.arange(pcd.shape[0])
                np.random.shuffle(index)
                index = index[0:5000]
                np.savetxt(kpsfn, index)

                # save keypoints' disparity
                disps = []
                for j in range(5000):
                    disps.append(disp[index[j]])
                disps = np.array(disps)
                np.save(f'{dispdir}/disp_{pc}.npy',disp)
                # save keypoints' pc
                kpcd = pcd[index]
                np.save(f'{kpcddir}/cloud_bin_{pc}Keypoints.npy',kpcd) """

    def save_kpdisp(self):
        for i in self.testseq:
            seq = self.kitti[f'{i}']
            kpsdir = f'{self.savedir}/{i}/Keypoints'
            make_non_exists_dir(kpsdir)
            dispdir = f'{self.savedir}/{i}/Disparity'
            make_non_exists_dir(dispdir)
            for pair, trans in tqdm(seq['pair'].items()):
                pc0,pc1=str.split(pair,'-')
                pc0 = int(pc0)
                pc1 = int(pc1)
                key0 = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc0}Keypoints.txt').astype(np.int64)
                key1 = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc1}Keypoints.txt').astype(np.int64)
                disp0 = np.load(f'{self.savedir}/{i}/PointCloud/disparity/disp_{pc0}.npy')
                disp1 = np.load(f'{self.savedir}/{i}/PointCloud/disparity/disp_{pc1}.npy')
                # save keypoints' disparity
                disps0 = []
                disps1 = []
                for j in range(5000):
                    disps0.append(disp0[key0[j]])
                    disps1.append(disp1[key1[j]])
                disps0 = np.array(disps0)
                disps1 = np.array(disps1)
                np.save(f'{dispdir}/disp_{pc0}.npy',disp0)
                np.save(f'{dispdir}/disp_{pc1}.npy',disp1)

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
            # disp = disp[index[0:40000]]
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
            seq = self.kitti[f'{i}']
            savedir = f'./data/YOHO_FCGF/Testset/kittistereo/{i}/FCGF_Input_Group_feature'
            make_non_exists_dir(savedir)
            for pc in tqdm(seq['pc']):
                feats = []
                # load pointcloud and keypoints
                xyz = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.npy')
                # key = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
                # key = xyz[key]
                key = np.load(f'{self.savedir}/{i}/Keypoints_PC/cloud_bin_{pc}Keypoints.npy')
                feats = self.generate_scan_gfeats(xyz, key)
                np.save(f'{savedir}/{pc}.npy', feats)
                          
if __name__=='__main__':
    generator = generate_kitti()
    generator.loadset()
    # generator.Disp2PC()
    # generator.generate_gt()
    # generator.gt_log()
    # generator.load_save_pc()
    # generator.generate_kps_new()
    # generator.save_kpdisp()
    # generator.trainval_list()
    
    # test
    generator.generate_test_gfeats()
    