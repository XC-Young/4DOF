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

class generate_robotcar:
  def __init__(self):
    self.basedir = f'/home/hdmap/yxcdata/03_Data/RobotCar/2014-12-05-15-42-07/stereo_rgb'
    self.savedir = f'./data/origin_data/robotcar'
    make_non_exists_dir(self.savedir)

    self.yohosavedir = f'./data/YOHO_FCGF/robotstereo'
    """ self.load_model()
    self.G = np.load(f'./group_related/Rotation_8.npy')
    self.knn=knn_module.KNN(1)
    self.batchsize = 64
    self.n_train_rs = 2 """

  def loadset(self):
    self.imgname = []
    pair_fn = glob.glob(f'{self.basedir}/Disp_kittisize/*.png')
    for fn in pair_fn:
        pair = str.split(fn,'/')[-1]
        pair = str.split(pair,'.')[0]
        self.imgname.append(pair)
        
  def ImgPick(self):
     self.pickimgname = []
     pickIMGfn = '/home/hdmap/yxcdata/03_Data/RobotCar/2014-12-05-15-42-07/stereo_rgb/Disp_picked'
     make_non_exists_dir(pickIMGfn)
     for name in tqdm(self.imgname):
        imgname = name
        depthimg = cv2.imread(f'{self.basedir}/Disp/{imgname}.png',cv2.IMREAD_UNCHANGED)
        depthimg = depthimg[0:800,0:1280]
        depth = []
        for row in range(depthimg.shape[0]):
            for col in range(depthimg.shape[1]):
                d = depthimg[row, col]
                d = d/255.0
                depth.append(d)
        depth = np.array(depth)
        d_max = np.max(depth)
        d_min = np.min(depth)
        if d_max-d_min<80:
           self.pickimgname.append(imgname)
           cv2.imwrite(f'{pickIMGfn}/{imgname}.png',depthimg)


  def Disp2PC(self):
    pc_save = f'{self.savedir}/2014-12-05-15-42-07-kittisize/PointCloud'
    make_non_exists_dir(pc_save)
    disp_save = f'{pc_save}/disparity'
    make_non_exists_dir(disp_save)
    l_intrinsics_path = '/home/hdmap/yxcdata/03_Data/04_Data_process/robotcar/robotcar-dataset-sdk/models/stereo_wide_left.txt'
    with open(l_intrinsics_path) as l_intrinsics_file:
      l_vals = [float(x) for x in next(l_intrinsics_file).split()]
      fu1 = float(l_vals[0])
      fv1 = float(l_vals[1])
      cu1 = float(l_vals[2])
      cv1 = float(l_vals[3])
    baseline = 0.239983
    # baseline = 0.119997
    for name in tqdm(self.imgname):
        imgname = name
        colorimg = cv2.imread(f'{self.basedir}/left/{imgname}.png')
        colorimg = colorimg[0:800,0:1280]
        depthimg = cv2.imread(f'{self.basedir}/Disp/{imgname}.png',cv2.IMREAD_UNCHANGED)
        depthimg = depthimg[0:800,0:1280]
        # Crop the size of color image to depth image directly
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
        o3d.io.write_point_cloud(f'{pc_save}/cloud_bin_{imgname}.ply',ply)
        # np.save(f'{pc_save}/cloud_bin_{imgname}.npy',points)
        np.save(f'{disp_save}/disp_{imgname}.npy',disps)

if __name__=='__main__':
    generator = generate_robotcar()
    generator.loadset()
    # generator.ImgPick()
    generator.Disp2PC()