"""
Evaluator class for the whole pipeline of YOHO-C/O containing:
(1) Given FCGF group feature from ./YOHO_testset.py.
(2) PartI: FCGF group feature-->YOHO-Desc                       extractor
(3)        YOHO-Desc-->inv-->matmul matcher-->pps               matcher
(4)        pps+YOHO-Desc-->coarse rotation                      extractor
(5)        YOHO-C                                               estimator
(6) PartII:pps+YOHO-Desc+coarse_rotation-->refined rotation     extractor
(9)        YOHO-O                                               estimator
or namely, tester.
"""


import os,sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
import utils.RR_cal as RR_cal
from utils.dataset import get_dataset
from utils.utils import transform_points
from utils.r_eval import compute_R_diff
from tests.extractor import name2extractor,extractor_dr_index
from tests.matcher import name2matcher
from tests.estimator import name2estimator

def make_non_exists_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Evaluator_PartI:
    def __init__(self,cfg,max_iter):
        self.max_iter=max_iter
        self.cfg=cfg
        self.extractor=name2extractor[self.cfg.extractor](self.cfg)
        self.matcher=name2matcher[self.cfg.matcher](self.cfg)
        self.drindex_extractor=extractor_dr_index(self.cfg)
        est=self.cfg.estimator
        if self.max_iter>500:
            est='yohoc_mul'
        self.estimator=name2estimator[est](self.cfg)

    def run_onescene(self,dataset):
        #extractor:
        if not dataset.name[0:4]=='3dLo':
            self.extractor.Extract(dataset)
        self.matcher.match(dataset)
        self.drindex_extractor.PartI_Rindex(dataset)
        self.estimator.ransac(dataset,self.max_iter)

    def Feature_match_Recall(self,dataset,ratio=0.05):
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        # Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'
        pair_fmrs=[]
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #match
            matches=np.load(f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match/{id0}-{id1}.npy')
            keys0=dataset.get_kps(id0)[matches[:,0],:]
            keys1=dataset.get_kps(id1)[matches[:,1],:]
            #gt
            gt=dataset.get_transform(id0,id1)
            #ratio
            keys1=transform_points(keys1,gt)
            dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
            pair_fmr=np.mean(dist<self.cfg.ok_match_dist_threshold) #ok ratio in one pair
            pair_fmrs.append(pair_fmr)                              
        pair_fmrs=np.array(pair_fmrs)                               #ok ratios in one scene
        FMR=np.mean(pair_fmrs>ratio)                                #FMR in one scene
        return FMR, pair_fmrs

    def eval(self):
        datasets=get_dataset(self.cfg,False)
        max_iter=1000
        FMRS=[]
        all_pair_fmrs=[]
        for scene,dataset in datasets.items():
            if scene=='wholesetname':continue
            self.run_onescene(dataset)
            print(f'eval the FMR result on {dataset.name}')
            FMR,pair_fmrs=self.Feature_match_Recall(dataset,ratio=self.cfg.fmr_ratio)
            FMRS.append(FMR)
            all_pair_fmrs.append(pair_fmrs)
        FMRS=np.array(FMRS)
        all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)

        #RR
        datasetname=datasets['wholesetname']
        if datasetname[0:5]=='kitti':
            RRs=[]
            whole_ok_num=0
            whole_all_num=0
            whole_rre=[]
            whole_rte=[]
            for name,dataset in datasets.items():
                if name=='wholesetname':
                    continue
                oknum=0
                wholenum=0
                for pair in dataset.pair_ids:
                    writer=open(f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match/YOHO_C/{self.cfg.max_iter}iters/pre_RRE&RTE.log','a')
                    id0,id1=pair
                    wholenum+=1
                    gt=dataset.get_transform(id0,id1)
                    pre=np.load(f'data/YOHO_FCGF/Testset/{dataset.name}/Match/YOHO_C/{self.cfg.max_iter}iters/{id0}-{id1}.npz')['trans']
                    tdiff = np.linalg.norm(pre[0:3,-1]-gt[0:3,-1])
                    Rdiff=compute_R_diff(gt[0:3,0:3],pre[0:3,0:3])
                    
                    if tdiff<=2 and Rdiff<=5:
                        oknum+=1
                        writer.write(f'{int(id0)}\t{int(id1)}\tSucceed!\n')
                        writer.write(f'RRE:{Rdiff}\tRTE:{tdiff}\n')
                        if Rdiff<5:
                            whole_rre.append(Rdiff)
                        if tdiff<2:
                            whole_rte.append(tdiff)
                    else:
                        writer.write(f'{int(id0)}\t{int(id1)}\tFailed...\n')
                        writer.write(f'RRE:{Rdiff}\tRTE:{tdiff}\n')
                    writer.close()
                RRs.append(oknum/wholenum)
                whole_ok_num+=oknum
                whole_all_num+=wholenum
            Mean_Registration_Recall = np.mean(np.array(RRs))
            rre = np.mean(np.array(whole_rre))
            rte = np.mean(np.array(whole_rte))
            #print and save:
            msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
            msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
                f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
                f'Mean_Registration_Recall {Mean_Registration_Recall}\n' \
                f'RRE {rre}\n' \
                f'RTE {rte}\n'
        else:
            Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(self.cfg,datasets,self.max_iter,yoho_sign='YOHO_C')
            #print and save:
            msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
            msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
                f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
                f'Mean_Registration_Recall {Mean_Registration_Recall}\n'

        with open('data/results.log','a') as f:
            f.write(msg)
        print(msg)
               
class Evaluator_PartII:
    def __init__(self,cfg,max_iter):
        self.max_iter=max_iter
        self.cfg=cfg
        self.extractor=name2extractor[self.cfg.extractor](self.cfg)
        self.matcher=name2matcher[self.cfg.matcher](self.cfg)
        self.estimator=name2estimator[self.cfg.estimator](self.cfg)
        self.drindex_extractor=extractor_dr_index(self.cfg)

    def run_onescene(self,dataset):
        #extractor:
        self.matcher.match(dataset)
        self.drindex_extractor.PartI_Rindex(dataset)
        self.extractor.PartII_R_pre(dataset)
        self.estimator.ransac(dataset,self.max_iter)

    def Feature_match_Recall(self,dataset,ratio=0.05):
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        # Keys_dir=f'{self.cfg.origin_data_dir}/{datasetname}/Keypoints_PC'
        pair_fmrs=[]
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #match
            matches=np.load(f'{self.cfg.output_cache_fn}/Testset/{dataset.name}/Match/{id0}-{id1}.npy')
            keys0=dataset.get_kps(id0)[matches[:,0],:]
            keys1=dataset.get_kps(id1)[matches[:,1],:]
            #gt
            gt=dataset.get_transform(id0,id1)
            #ratio
            keys1=transform_points(keys1,gt)
            dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
            pair_fmr=np.mean(dist<self.cfg.ok_match_dist_threshold) #ok ratio in one pair
            pair_fmrs.append(pair_fmr)                              
        pair_fmrs=np.array(pair_fmrs)                               #ok ratios in one scene
        FMR=np.mean(pair_fmrs>ratio)                                #FMR in one scene
        return FMR, pair_fmrs

    def eval(self):
        datasets=get_dataset(self.cfg,False)
        FMRS=[]
        all_pair_fmrs=[]
        for scene,dataset in datasets.items():
            if scene=='wholesetname':continue
            self.run_onescene(dataset)
            print(f'eval the FMR result on {dataset.name}')
            FMR,pair_fmrs=self.Feature_match_Recall(dataset,ratio=self.cfg.fmr_ratio)
            FMRS.append(FMR)
            all_pair_fmrs.append(pair_fmrs)
        FMRS=np.array(FMRS)
        all_pair_fmrs=np.concatenate(all_pair_fmrs,axis=0)

        #RR
        datasetname=datasets['wholesetname']
        if datasetname[0:5]=='kitti':
            RRs=[]
            whole_ok_num=0
            whole_all_num=0
            whole_rre=[]
            whole_rte=[]
            for name,dataset in datasets.items():
                if name=='wholesetname':
                    continue
                oknum=0
                wholenum=0
                for pair in dataset.pair_ids:
                    id0,id1=pair
                    wholenum+=1
                    gt=dataset.get_transform(id0,id1)
                    pre=np.load(f'data/YOHO_FCGF/Testset/{dataset.name}/Match/YOHO_O/{self.cfg.max_iter}iters/{id0}-{id1}.npz')['trans']
                    tdiff = np.linalg.norm(pre[0:3,-1]-gt[0:3,-1])
                    Rdiff=compute_R_diff(gt[0:3,0:3],pre[0:3,0:3])
                    if tdiff<=2 and Rdiff<=5:
                        oknum+=1
                        if Rdiff<5:
                            whole_rre.append(Rdiff)
                        if tdiff<2:
                            whole_rte.append(tdiff)
                RRs.append(oknum/wholenum)
                whole_ok_num+=oknum
                whole_all_num+=wholenum
            Mean_Registration_Recall = np.mean(np.array(RRs))
            rre = np.mean(np.array(whole_rre))
            rte = np.mean(np.array(whole_rte))
            #print and save:
            msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
            msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
                f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
                f'Mean_Registration_Recall {Mean_Registration_Recall}\n' \
                f'RRE {rre}\n' \
                f'RTE {rte}\n'
        else:
            Mean_Registration_Recall,c_flags,c_errors=RR_cal.benchmark(self.cfg,datasets,self.max_iter,yoho_sign='YOHO_O')
            #print and save:
            msg=f'{datasetname}-{self.cfg.descriptor}-{self.cfg.extractor}-{self.cfg.matcher}-{self.cfg.estimator}-{self.max_iter}iterations\n'
            msg+=f'correct ratio avg {np.mean(all_pair_fmrs):.5f}\n' \
                f'correct ratio>0.05 avg {np.mean(FMRS):.5f}  std {np.std(FMRS):.5f}\n' \
                f'Mean_Registration_Recall {Mean_Registration_Recall}\n'

        with open('data/results.log','a') as f:
            f.write(msg)
        print(msg)


name2evaluator={
    'PartI':Evaluator_PartI,
    'PartII':Evaluator_PartII
}