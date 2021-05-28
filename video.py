import numpy as np
from tqdm import tqdm
import skvideo.io
def shrink(X,tau):
    Y = np.abs(X)-tau
    return np.sign(X) * np.maximum(Y,np.zeros_like(Y))
def SVT(X,tau):
    U,S,VT = np.linalg.svd(X,full_matrices=0)
    out = U @ np.diag(shrink(S,tau)) @ VT
    return out
def RPCA(X,max_iter):
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    lambd = 1/np.sqrt(np.maximum(n1,n2))
    thresh = 10**(-7) * np.linalg.norm(X,ord='fro')
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)

    for i in tqdm(range(max_iter)):
        if (np.linalg.norm(X-L-S,ord='fro') <= thresh):
            break
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
    return L,S

def run():
    video_data = skvideo.io.vread('./data/Video_240p.mp4',as_grey=True)
    frames = video_data[...,0]
    X = frames.reshape((frames.shape[0],-1))
    L,S = RPCA(X,1000)

    L = L.reshape((L.shape[0],240,320))
    S = S.reshape((L.shape[0],240,320))
    skvideo.io.vwrite("./data/back.mp4",L)
    skvideo.io.vwrite("./data/front.mp4",S)
if __name__ == '__main__':
    run()