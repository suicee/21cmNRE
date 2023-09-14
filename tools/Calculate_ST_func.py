import torch
import numpy as np
from kymatio.torch import Scattering2D,HarmonicScattering3D


def abs_log(x):
    re=(np.sign(x))*np.log2(np.abs(x))
    re[np.isnan(re)]=0
    return re


def calculate_ST_total(lc,scattering):

    lc=lc-np.mean(lc,axis=(0,1))
    cube=torch.tensor(lc,dtype=torch.float32)
    cube=cube.cuda()
    sc=scattering(cube)
    sc=(np.mean(abs_log(sc.cpu().numpy()),axis=1)).flat
    sc0=[(torch.sum(cube**i)).cpu().numpy() for i in [2,3,4]]
    sc0=abs_log(np.array(sc0))
    total_sc=np.hstack((sc0,sc))

    return total_sc



def calculate_ST_3D(lc,scattering):
    #remove mean value(slice by slice) to mimic real observation
    lc=lc-np.mean(lc,axis=(0,1))
    #break lightcone into chunks(batches)
    cube=np.array(np.split(lc.transpose(2,0,1),10))
    cube=torch.tensor(cube,dtype=torch.float32)
    cube=cube.cuda()
    #calculate 1st and 2rd st coefs
    sc=scattering(cube)
    sc=(np.mean(abs_log(sc.cpu().numpy()),axis=2)).reshape(10,-1)
    #calculate 0th coefs
    sc0=[(torch.sum(cube**i,dim=(1,2,3))).cpu().numpy() for i in [2,3,4]]
    sc0=abs_log(np.array(sc0).transpose())
    #combine all coefs
    total_sc=np.hstack((sc0,sc))
    #flip to start from high redshift
    total_sc=np.flip(total_sc,axis=0)

    return total_sc




