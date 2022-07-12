import os
import torch
import numpy as np
import glob, random
from utils.utils import *
from utils.r_eval import *
import open3d as o3d
from tqdm import tqdm
from fcgf_model import load_model
from knn_search import knn_module
from utils.misc import extract_features

class generate_kitti:
    def __init__(self):
        self.trainseq = [0,1,2,3,4,5]
        self.valseq = [6,7]
        self.testseq = [8,9,10]
        self.basedir = f'/home/hdmap/yxcdata/YOHO/data/origin_data/KITTI'
        self.savedir = f'./data/origin_data/kitti'
        self.yohosavedir = f'./data/YOHO_FCGF/kitti'
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
        # for j in range(8,10):
        #     f = open(f'/home/hdmap/yxcdata/Data_process/kitti_gt/test/{j}.pkl','rb')
        #     data = pickle.load(f)
        #     for k in range(len(data)):
        #         matedata = data[k]
        #         trans=matedata['transform']
        #         m=matedata['frame0']
        #         n=matedata['frame1']
    
    def gt_log(self):
        for key, val in self.kitti.items():
            fn = f'{self.savedir}/{key}/PointCloud/gt.log'
            make_non_exists_dir(f'{self.savedir}/{key}/PointCloud')
            writer=open(fn,'w')
            pc_num = len(val['pc'])
            for pair, transform_pr in val['pair'].items():
                pc0,pc1=str.split(pair,'-')
                transform_pr = np.linalg.inv(transform_pr)
                writer.write(f'{int(pc0)}\t{int(pc1)}\t{pc_num}\n')
                writer.write(f'{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n')
                writer.write(f'{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n')
                writer.write(f'{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n')
                writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
        writer.close()
        
    def load_save_pc(self):
        for key, val in self.kitti.items():
            key = int(key)
            plydir = f'{self.savedir}/{key}/PointCloud'
            key = f'{key}'.zfill(2)
            bindir = f'{self.basedir}/dataset/sequences/{key}/velodyne'
            for p in tqdm(val['pc']):
                p = int(p)
                p6 = f'{p}'.zfill(6)
                binfn = f'{bindir}/{p6}.bin'
                pcd = np.fromfile(binfn, dtype=np.float32).reshape(-1, 4)[:,0:3]
                ply = o3d.geometry.PointCloud()
                ply.points = o3d.utility.Vector3dVector(pcd)
                o3d.io.write_point_cloud(f'{plydir}/cloud_bin_{p}.ply', ply)
                np.save(f'{plydir}/cloud_bin_{p}.npy',pcd)

    def generate_kps(self):
        for key, val in self.kitti.items():
            kpsdir = f'{self.savedir}/{key}/Keypoints'
            make_non_exists_dir(kpsdir)
            for p in tqdm(val['pc']):
                p = int(p)
                kpsfn = f'{kpsdir}/cloud_bin_{p}Keypoints.txt'
                # os.system(f'mv {self.savedir}/{key}/PointCloud/{p}.ply {self.savedir}/{key}/PointCloud/cloud_bin_{p}.ply')
                ply = o3d.io.read_point_cloud(f'{self.savedir}/{key}/PointCloud/cloud_bin_{p}.ply')
                pcd = np.array(ply.points)
                index = np.arange(pcd.shape[0])
                np.random.shuffle(index)
                index = index[0:5000]
                np.savetxt(kpsfn, index)
                
    def gt_match(self):
        for seqs in [self.trainseq, self.valseq]:
            for i in seqs:
                seq = self.kitti[f'{i}']
                savedir = f'{self.yohosavedir}/{i}/gt_match'
                make_non_exists_dir(savedir)
                for pair,trans in tqdm(seq['pair'].items()):
                    id0,id1=str.split(pair,'-')
                    pc0 = o3d.io.read_point_cloud(f'{self.savedir}/{i}/PointCloud/cloud_bin_{id0}.ply')
                    pc1 = o3d.io.read_point_cloud(f'{self.savedir}/{i}/PointCloud/cloud_bin_{id1}.ply')
                    pc0 = np.array(pc0.points)
                    pc1 = np.array(pc1.points)
                    key0 = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{id0}Keypoints.txt').astype(np.int)
                    key1 = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{id1}Keypoints.txt').astype(np.int)
                    key0 = pc0[key0]
                    key1 = pc1[key1]
                    key0 = self.apply_transform(key0, trans)
                    dist = np.sum(np.square(key0[:,None,:]-key1[None,:,:]),axis=-1)
                    # match
                    thres = 0.3*1.5
                    d_min = np.min(dist,axis=1)
                    arg_min = np.argmin(dist,axis=1)
                    m0 = np.arange(d_min.shape[0])[d_min<thres*thres]
                    m1 = arg_min[d_min<thres*thres]
                    pair = np.concatenate([m0[:,None],m1[:,None]],axis=1)
                    save_fn = f'{savedir}/{id0}_{id1}.npy'
                    np.save(save_fn, pair)

    def load_model(self):
        checkpoint = torch.load('./model/Backbone_large/best_val_checkpoint.pth')
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

    def apply_random_rots(self):
        # two random rotations
        for i in self.trainseq:
            seq = self.kitti[f'{i}']
            savedir = f'{self.yohosavedir}/{i}/rot_feats'
            make_non_exists_dir(savedir)
            for pc in tqdm(seq['pc']):
                rot = np.concatenate([random_rotation_matrix()[None,:,:] for r in range(self.n_train_rs)],axis=0)
                np.save(f'{savedir}/{pc}_Rs.npy', rot)
                    
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
        feats = np.concatenate(feats, axis=-1)
        return feats

    def generate_gfeats(self):
        for i in self.trainseq:
            seq = self.kitti[f'{i}']
            savedir = f'{self.yohosavedir}/{i}/rot_feats'
            make_non_exists_dir(savedir)
            for pc in tqdm(seq['pc']):
                feats = []
                # load pointcloud and keypoints
                xyz = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.npy')
                key = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
                key = xyz[key]
                # apply rotations
                rs = np.load(f'{savedir}/{pc}_Rs.npy')
                for rid in range(rs.shape[0]):
                    r = rs[rid]
                    # rot pc and keys
                    xyzr = xyz@r.T
                    keyr = key@r.T
                    featr = self.generate_scan_gfeats(xyzr, keyr)
                    feats.append(featr[None,:,:,:])
                feats = np.concatenate(feats,axis=0)
                np.save(f'{savedir}/{pc}_feats.npy', feats)

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.G.shape[0]):
            R_diff=compute_R_diff(self.G[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id

    def DeltaR(self,R,index):
        R_anchor=self.G[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)

    def generate_batches(self, start = 0):
        batchsavedir = f'{self.yohosavedir}/trainset'
        make_non_exists_dir(batchsavedir)
        batch_i = start
        for i in self.trainseq:
            seq = self.kitti[f'{i}']
            savedir = f'{self.yohosavedir}/{i}/rot_feats'
            make_non_exists_dir(savedir)
            for pair, trans in tqdm(seq['pair'].items()):
                id0,id1=str.split(pair,'-')
                Rs0 = np.load(f'{self.yohosavedir}/{i}/rot_feats/{id0}_Rs.npy')
                Rs1 = np.load(f'{self.yohosavedir}/{i}/rot_feats/{id1}_Rs.npy')
                feats0 = np.load(f'{self.yohosavedir}/{i}/rot_feats/{id0}_feats.npy')
                feats1 = np.load(f'{self.yohosavedir}/{i}/rot_feats/{id1}_feats.npy')
                pair = np.load(f'{self.yohosavedir}/{i}/gt_match/{id0}_{id1}.npy').astype(np.int32)
                index = np.arange(pair.shape[0])
                np.random.shuffle(index)
                index = index[0:self.batchsize]
                pair = pair[index]
                # paired feats
                feats0 = feats0[:,pair[:,0],:,:]
                feats1 = feats1[:,pair[:,1],:,:]
                # gt
                Rgt = np.load(f'{self.basedir}/icp/{i}_{id0}_{id1}.npy')[0:3,0:3]
                # here we should randomly choose (0,1) of the rotations
                chosenr_m0 = np.random.choice(self.n_train_rs,size = self.batchsize)
                chosenr_m1 = np.random.choice(self.n_train_rs,size = self.batchsize)
                chosenr_R0 = Rs0[chosenr_m0]
                chosenr_R1 = Rs1[chosenr_m1]
                chosenr_feat0 = feats0[chosenr_m0,np.arange(self.batchsize)]
                chosenr_feat1 = feats1[chosenr_m1,np.arange(self.batchsize)]
                # get ground truth R, Rindex, residualR
                batch_Rs, batch_Rids, batch_Rres = [],[],[]
                for b in range(self.batchsize):
                    R = chosenr_R1[b] @ Rgt @ chosenr_R0[b].T
                    Rid = self.R2DR_id(R)
                    Rres = self.DeltaR(R, Rid)
                    batch_Rs.append(R[None,:,:])
                    batch_Rids.append(Rid)
                    batch_Rres.append(Rres[None,:])
                batch_Rs = np.concatenate(batch_Rs, axis=0)
                batch_Rids = np.array(batch_Rids)
                batch_Rres = np.concatenate(batch_Rres, axis=0)
                # joint to be a batch
                item={
                        'feats0':torch.from_numpy(chosenr_feat0.astype(np.float32)), #before enhanced rot
                        'feats1':torch.from_numpy(chosenr_feat1.astype(np.float32)), #after enhanced rot
                        'R':torch.from_numpy(batch_Rs.astype(np.float32)),
                        'true_idx':torch.from_numpy(batch_Rids.astype(np.int)),
                        'deltaR':torch.from_numpy(batch_Rres.astype(np.float32))
                    }
                # save
                torch.save(item,f'{batchsavedir}/{batch_i}.pth',_use_new_zipfile_serialization=False)
                batch_i += 1

    def generate_val_gfeats(self):
        for i in self.valseq:
            seq = self.kitti[f'{i}']
            savedir = f'{self.yohosavedir}/{i}/rot_feats'
            make_non_exists_dir(savedir)
            for pc in tqdm(seq['pc']):
                feats = []
                # load pointcloud and keypoints
                xyz = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.npy')
                key = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
                key = xyz[key]
                # apply rotations
                r = random_rotation_matrix()
                np.save(f'{savedir}/{pc}_Rs.npy',r)
                # rot pc and keys
                xyzr = xyz@r.T
                keyr = key@r.T
                featr = self.generate_scan_gfeats(xyzr, keyr)
                np.save(f'{savedir}/{pc}_feats.npy', featr)
                
    def generate_val_batches(self, vallen = 3000):
        batchsavedir = f'{self.yohosavedir}/valset'
        make_non_exists_dir(batchsavedir)
        # generate matches
        matches = []
        for i in self.valseq:
            seq = self.kitti[f'{i}']
            savedir = f'{self.yohosavedir}/{i}/rot_feats'
            make_non_exists_dir(savedir)
            for pair, trans in tqdm(seq['pair'].items()):
                id0,id1=str.split(pair,'-')
                pair = np.load(f'{self.yohosavedir}/{i}/gt_match/{id0}_{id1}.npy').astype(np.int32)
                for p in range(pair.shape[0]):
                    matches.append((i,id0,id1,pair[p][0],pair[p][1]))
        random.shuffle(matches)        
        batch_i = 0
        for batch_i in tqdm(range(vallen)):
            tup = matches[batch_i]
            scene, id0, id1, pt0, pt1 = tup
            R0 = np.load(f'{self.yohosavedir}/{scene}/rot_feats/{id0}_Rs.npy')
            R1 = np.load(f'{self.yohosavedir}/{scene}/rot_feats/{id1}_Rs.npy')
            feat0 = np.load(f'{self.yohosavedir}/{scene}/rot_feats/{id0}_feats.npy')[int(pt0)]
            feat1 = np.load(f'{self.yohosavedir}/{scene}/rot_feats/{id1}_feats.npy')[int(pt1)]
            # gt
            Rgt = np.load(f'{self.basedir}/icp/{scene}_{id0}_{id1}.npy')[0:3,0:3]
            # get ground truth R, Rindex, residualR
            Rbase = R1 @ Rgt @ R0.T
            Rid = self.R2DR_id(Rbase)
            Rres = self.DeltaR(Rbase, Rid)
            
            # joint to be a batch
            item={
                    'feats0':torch.from_numpy(feat0.astype(np.float32)), #before enhanced rot
                    'feats1':torch.from_numpy(feat1.astype(np.float32)), #after enhanced rot
                    'R':torch.from_numpy(Rbase.astype(np.float32)),
                    'true_idx':torch.from_numpy(np.array([Rid]).astype(np.int)),
                    'deltaR':torch.from_numpy(Rres.astype(np.float32))
                }
            # save
            torch.save(item,f'{batchsavedir}/{batch_i}.pth',_use_new_zipfile_serialization=False)
            batch_i += 1
                
    def trainval_list(self):
        traindir = f'{self.yohosavedir}/trainset'
        valdir = f'{self.yohosavedir}/valset'
        trainlist = glob.glob(f'{traindir}/*.pth')
        vallist = glob.glob(f'{valdir}/*.pth')
        save_pickle(range(len(trainlist)), f'{self.yohosavedir}/train.pkl')
        save_pickle(range(len(vallist)), f'{self.yohosavedir}/val.pkl')      

    # def generate_test_gfeats(self):
    #     for i in self.testseq:
    #         seq = self.kitti[f'{i}']
    #         savedir = f'./data/YOHO_FCGF/Testset/kitti/{i}/FCGF_Input_Group_feature'
    #         make_non_exists_dir(savedir)
    #         for pc in tqdm(seq['pc']):
    #             feats = []
    #             # load pointcloud and keypoints
    #             xyz = np.load(f'{self.savedir}/{i}/PointCloud/cloud_bin_{pc}.npy')
    #             key = np.loadtxt(f'{self.savedir}/{i}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
    #             key = xyz[key]
    #             feats = self.generate_scan_gfeats(xyz, key)
    #             np.save(f'{savedir}/{pc}.npy', feats)
                          
    def generate_test_gfeats(self):
        for i in self.testseq:
            testdir = f'./data/origin_data/kitti_test/{i}'
            savedir = f'./data/YOHO_FCGF/Testset/kitti/{i}/FCGF_Input_Group_feature'
            make_non_exists_dir(savedir)
            num = os.listdir(f'{testdir}/Keypoints/')
            for pc in tqdm(range(len(num))):
                feats = []
                # load pointcloud and keypoints
                ply = o3d.io.read_point_cloud(f'{testdir}/PointCloud/cloud_bin_{pc}.ply')
                xyz = np.asarray(ply.points)
                key = np.loadtxt(f'{testdir}/Keypoints/cloud_bin_{pc}Keypoints.txt').astype(np.int64)
                key = xyz[key]
                feats = self.generate_scan_gfeats(xyz, key)
                np.save(f'{savedir}/{pc}.npy', feats)

if __name__=='__main__':
    generator = generate_kitti()
    # generator.loadset()
    # generator.gt_log()
    # generator.load_save_pc()
    # generator.generate_kps()
    # generator.gt_match()
    # generator.apply_random_rots()
    # generator.generate_gfeats()
    # # generate more batches
    # for i in range(2):
    #     generator.generate_batches(start = len(glob.glob(f'{generator.yohosavedir}/trainset/*.pth')))
    # generator.generate_val_gfeats()
    # generator.generate_val_batches()
    # generator.trainval_list()
    
    #test
    generator.generate_test_gfeats()
    

        # f=open('2.txt','a')
        # print(self.kitti[f'{7}'],file=f)
        # f.close()
