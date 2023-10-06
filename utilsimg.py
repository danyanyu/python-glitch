import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import prange, njit
from skimage import io
from time import time

gausmatrix = np.array([[2,4,5,4,2],
                       [4,9,12,9,4],
                       [5,12,15,12,5],
                       [4,9,12,9,4],
                       [2,4,5,4,2]])/159
gaus1d = np.exp(-0.5*(np.arange(-2,3))**2)/np.sqrt(2*np.pi)

@njit
def togray(p):
    c = np.sqrt((p[:,:,0]/255)**2 + (p[:,:,1]/255)**2 + (p[:,:,2]/255)**2)
    return c

@njit
def togray2(p):
    x,y,_=p.shape
    k=np.zeros((x,y,3))
    c = (p[:,:,0]+p[:,:,1]+p[:,:,2])/3
    k[:,:,0]=c
    k[:,:,1]=c
    k[:,:,2]=c
    return k

@njit
def gausblur5(p):
    c = p.copy()
    first, second = c.shape
    for i in prange(2, first-2):
        for j in range(2, second-2):
            c[i, j] = np.average(c[i-2:i+3,j-2:j+3],weights=gausmatrix)
    return c

@njit
def gausblurfast(p):
    c = p.copy()
    first, second = c.shape[0],c.shape[1]
    for i in prange(2, first-2):
        for j in range(2, second-2):
            c[i, j] = np.average(p[i-2:i+3,j],weights=gaus1d)
    for i in prange(2, first-2):
        for j in range(2, second-2):
            c[i, j] = np.average(p[i,j-2:j+3],weights=gaus1d)
    return c

@njit
def gx(p):
    c = np.zeros_like(p)
    first, second = c.shape
    coef = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    for i in prange(1, first-1):
        for j in range(1, second-1):
            c[i,j] += 1*(p[i-1,j-1]+p[i+1,j-1])
            c[i,j] += -1*(p[i-1,j+1]+p[i+1,j+1])
            c[i,j] += 2*p[i,j-1]
            c[i,j] += -2*p[i,j+1]
    return c

@njit
def gy(p):
    c = np.zeros_like(p)
    first, second = c.shape
    coef = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    for i in prange(1, first-1):
        for j in range(1, second-1):
            c[i,j] += 1*(p[i-1,j-1]+p[i-1,j+1])
            c[i,j] += -1*(p[i+1,j-1]+p[i+1,j+1])
            c[i,j] += 2*p[i-1,j]
            c[i,j] += -2*p[i+1,j]
    return c

@njit
def thresh(p):
    c = np.arctan2(gy(p),gx(p))
    m = np.sqrt(gx(p)**2 + gy(p)**2)
    k = np.zeros_like(p)
    first, second = c.shape
    for i in prange(1,first-1):
        for j in range(1,second-1):
            if np.pi/(-8)<=c[i,j]<=np.pi/8 or c[i,j]>=np.pi*7/8 or c[i,j]<=np.pi*7/(-8):
                if m[i,j-1]<=m[i,j] and m[i,j]>=m[i,j+1]:#0
                    k[i,j]=m[i,j] 
            elif np.pi/8<=c[i,j]<=np.pi*3/8 or np.pi*5/(-8)>=c[i,j]>=np.pi*7/(-8):
                if m[i-1,j+1]<=m[i,j] and m[i,j]>=m[i+1,j-1]:#45
                    k[i,j]=m[i,j] 
            elif np.pi*3/8<=c[i,j]<=np.pi*5/8 or np.pi*3/8<=(-1)*c[i,j]<=np.pi*5/8:
                if m[i-1,j]<=m[i,j] and m[i,j]>=m[i+1,j]:#90
                    k[i,j]=m[i,j] 
            else:
                if m[i-1,j-1]<=m[i,j] and m[i,j]>=m[i+1,j+1]:#135
                    k[i,j]=m[i,j]           
    return k

@njit
def dfresh(p, low=0.05,high=0.08):
    weak = np.zeros_like(p)
    strong = np.zeros_like(p)
    first, second = p.shape
    m = np.max(p)
    for i in prange(first):
        for j in range(second):
            if p[i,j] >= high*m:
                strong[i,j]=1
            elif p[i,j] >=low*m:
                weak[i,j]=1
    for i in prange(1,first-1):
        for j in range(1,second-1):
            if weak[i,j] > 0:
                if np.sum(strong[i-1:i+2,j-1:j+2])==0:
                    weak[i,j]=0
    return (strong + weak)*(-1)+1

#noise = io.imread("images/noise.png")[:,:,0]/255

def stippling(p):
    c = np.zeros_like(p)
    yd, xd = p.shape
    xn = xd//32
    yn = yd//32
    xr = xd%32
    yr = yd%32
    for i in range(yn):
        for j in range(xn):
            for ii in range(32):
                for jj in range(32):
                    if p[i*32+ii,j*32+jj]>noise[ii,jj]:
                        c[i*32+ii,j*32+jj] = 1
    for j in range(xn):
        for ii in range(yr):
            for jj in range(32):
                if p[yn*32+ii,j*32+jj]>noise[ii,jj]:
                    c[yn*32+ii,j*32+jj] = 1
    for i in range(yn):
        for jj in range(xr):
            for ii in range(32):
                if p[i*32+ii,xn*32+jj]>noise[ii,jj]:
                    c[i*32+ii,xn*32+jj] = 1
    for ii in range(yr):
        for jj in range(xr):
            if p[yn*32+ii,xn*32+jj]>noise[ii,jj]:
                    c[yn*32+ii,xn*32+jj] = 1
    return c

def combine(p1,edges):
    res = np.ones_like(p1)
    first, second = p1.shape
    for i in range(first):
        for j in range(second):
            if edges[i,j]>0:
                res[i,j]=p1[i,j]
            else:
                res[i,j]=0
    return res