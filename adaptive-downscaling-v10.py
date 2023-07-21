import cv2
import numpy as np
import os
from tqdm import tqdm
import math
from typing import Union, Tuple
import fast_glcm
import time

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.morphology import square
from sklearn.cluster import KMeans

import argparse
import csv


# Ver. 7 (Refactored)
# - apply blurring before GLCM
# Ver. 8
# - amplifying line component
# Ver. 9
# - mitigate bleed-through
# - refactored

# Ver. 7 (Refactored)
# - apply blurring before GLCM
# Ver. 8
# - amplifying line component
# Ver. 9
# - mitigate bleed-through
# ver. 10
# - added more edge options. e.g., scharr, canny

class ContentsAwareAdaptiveDownscaling:
    def __init__(self, debug:bool=False, vis:bool=False) -> None:
        # Vars for original input
        self.ori_image = None
        self.h_old = None
        self.w_old = None
        self.ratio_ori = None
        self.ori_image_avg = None
        self.ori_image_std = None
        # Vars for target output
        self.size = None
        self.h_new = None
        self.w_new = None
        self.resized_image = None
        self.alpha     = []
        self.alpha_std = []
        self.std_th1 = None
        self.std_th2 = None
        # Vars for resizing
        self.h_ratio = None
        self.w_ratio = None
        self.scope = None
        self.buffer = []

        # Params: Debugging
        self.DEBUG = debug
        
        # Params: Adaptive Sampling
        self.map = None
        self.entr_scale = None
        self.cond_threshold = None
        self.scope_scale = None
        
        # Params: Visualization
        self.VIS = vis
        self.vis_radius = 0
        self.vis_red = (255,0,0)
        self.vis_blue = (0,0,255)
        self.vis_green = (0,255,0)
        self.vis_thickness = -1
        self.patch_image_viz_uniform_samples  = None
        self.patch_image_viz_adaptive_samples = None
        self.patch_image_viz_samples          = None

    '''
    _compute_std_thresholds
    '''
    def _compute_std_thresholds(self) -> None:
        # get the start time
        st = time.time()

        _patch_stds = [] # 7x7 patches
        _height,_width = self.ori_image.shape[:2]
        # collect random patches
        for i in range(10000):
            r,c = np.random.randint(3,_height-3),np.random.randint(3,_width-3)
            _patch = self.ori_image[r-3:r+4,c-3:c+4]
            assert _patch.shape == (7,7)
            _patch_stds.append(np.std(_patch).astype(np.float64))
        # compute std thresholds using KMeans
        _patch_stds = np.array(_patch_stds,dtype=np.float64).reshape(-1,1)
        _kmeans = KMeans(n_clusters=3, random_state=0).fit(_patch_stds)
        g1_mean, g2_mean, g3_mean = sorted(_kmeans.cluster_centers_.squeeze())
        g1_index = np.argwhere(_kmeans.cluster_centers_.squeeze()==g1_mean)[0][0]
        g2_index = np.argwhere(_kmeans.cluster_centers_.squeeze()==g2_mean)[0][0]
        g3_index = np.argwhere(_kmeans.cluster_centers_.squeeze()==g3_mean)[0][0]
        g1_prop = len(_kmeans.labels_[_kmeans.labels_==g1_index])/len(_kmeans.labels_)
        g2_prop = len(_kmeans.labels_[_kmeans.labels_==g2_index])/len(_kmeans.labels_)
        g3_prop = len(_kmeans.labels_[_kmeans.labels_==g3_index])/len(_kmeans.labels_)
        _th1 = g1_mean + (g2_mean - g1_mean)*(g2_prop/(g1_prop+g2_prop))
        _th2 = g2_mean + (g3_mean - g2_mean)*(g3_prop/(g2_prop+g3_prop))
        self.std_th1 = _th1
        self.std_th2 = _th2

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st
        print('precomputation is done: {} seconds'.format(elapsed_time))
        #exit()

    '''
    _set_input
    '''
    def _set_input(self, in_data) -> None:
        if(type(in_data) == str):
            self.ori_image = cv2.imread(in_data,0)
        else:
            # make sure input image is grayscale
            assert len(in_data.shape)==2
            self.ori_image = in_data
        self.h_old, self.w_old = self.ori_image.shape[:2]
        self.ratio_ori = np.divide(self.w_old, self.h_old)
        self.ori_image_avg = np.mean(self.ori_image)
        self.ori_image_std = np.std(self.ori_image)
        # Approximate two thresholds for the standard deviation of the original image
        self._compute_std_thresholds()
        
    '''
    _set_target_shape_by_size
    '''
    def _set_target_shape_by_size(self, size:int) -> None:
        # Set new size
        self.size = size
        # Set new height and width accordingly
        _h_new = np.sqrt(np.divide(self.size, self.ratio_ori))
        _w_new  = np.divide(self.size, _h_new)
        self.h_new = int(_h_new)
        self.w_new  = int(_w_new)
    
    '''
    resize
    '''
    def resize(self,
               in_data        : Union[str,np.ndarray], 
               size           : Union[Tuple[int,int],int]=720000,
               mode           : str='adaptive', 
               kernel_scale   : int=3, 
               cond_threshold : int=128, 
               scope_scale    : int=2, 
               opt            : str=None,
               amp            : bool=False) -> None:
        
        # Set input: (1) read image, (2) store shape, size, etc.
        self._set_input(in_data)
        
        # Set target shape either by (h,w) or size
        if type(size) == tuple:
            self.h_new, self.w_new = size
        else:
            self._set_target_shape_by_size(size)
        
        # Initialize output 
        self.resized_image = (np.ones([self.h_new, self.w_new])*255).astype(np.uint8)
            
        # Initialize mapping ratio
        self.w_ratio = float(self.w_old - 1) / (self.w_new - 1) if self.w_new > 1 else 0
        self.h_ratio = float(self.h_old - 1) / (self.h_new - 1) if self.h_new > 1 else 0
        
        # Start main process
        if(mode=='uniform'):
            if self.DEBUG: print("Uniform Resizing")
            self.resized_image = cv2.resize(self.ori_image, (self.w_new,self.h_new), cv2.INTER_LANCZOS4)
            
        elif(mode=='adaptive'):
            # Set parameters
            self.kernel_scale = kernel_scale
            self.cond_threshold = cond_threshold
            self.scope_scale = scope_scale
                
            if self.DEBUG: print("Adaptive Resizing")
            # Resize image based on either "LoG" or "GLCM"
            if(opt=='log'):
                #cleaned_image = skimage.exposure.match_histograms(self.ori_image, cv2.imread('../../data/00674590.png',0)).astype(np.uint8)
                self.map = cv2.Laplacian(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), cv2.CV_16S, ksize=kernel_scale)
            elif(opt=='dissimilarity'):
                self.map = fast_glcm.fast_glcm_dissimilarity(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
            elif(opt=='contrast'):
                self.map = fast_glcm.fast_glcm_contrast(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
            elif(opt=='homogeneity'):
                self.map = fast_glcm.fast_glcm_homogeneity(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
            elif(opt=='entropy'):
                self.map = fast_glcm.fast_glcm_entropy(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
                #self.map = fast_glcm.fast_glcm_entropy(self.ori_image, ks=kernel_scale)
            elif(opt=='max'):
                self.map = fast_glcm.fast_glcm_max(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
            elif(opt=='ASM'):
                self.map,_ = fast_glcm.fast_glcm_ASM(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
            elif(opt=='energy'):
                _,self.map = fast_glcm.fast_glcm_ASM(cv2.GaussianBlur(self.ori_image,(kernel_scale,kernel_scale),cv2.BORDER_DEFAULT), ks=kernel_scale)
            elif(opt=='scharr'):
                scharrx = cv2.Scharr(self.ori_image, cv2.CV_64F, 1, 0)
                scharry = cv2.Scharr(self.ori_image, cv2.CV_64F, 0, 1)
                self.map = np.sqrt(np.power(scharrx,2) + np.power(scharrx,2)) 
            elif(opt=='canny'):
                # Setting parameter values
                t_lower = 210  # Lower Threshold
                t_upper = 260  # Upper threshold
                # Applying the Canny Edge filter
                self.map = cv2.Canny(self.ori_image, t_lower, t_upper)    
            else:
                print("Unexpected resize option.")
                return
            
            for r in tqdm(range(self.h_new)):
                for c in range(self.w_new):
                    # (Step 4.1)
                    c_l, r_l = math.floor(self.w_ratio * c), math.floor(self.h_ratio * r)
                    c_h, r_h = math.ceil(self.w_ratio * c), math.ceil(self.h_ratio * r)
                    # Corner case handling
                    if(c_h >= self.w_old):
                        c_h = self.w_old-1
                    if(r_h >= self.h_old):
                        r_h = self.h_old-1
                        
                    r_TL = None
                    c_TL = None
                    r_TR = None
                    c_TR = None
                    r_BR = None
                    c_BR = None
                    r_BL = None
                    c_BL = None
                    
                    stat_TL = {'std':self.ori_image_std}
                    stat_TR = {'std':self.ori_image_std}
                    stat_BR = {'std':self.ori_image_std}
                    stat_BL = {'std':self.ori_image_std}
                    '''
                    Top-Left Point
                    '''
                    if(self.ori_image[r_l,c_l]<self.cond_threshold):
                        if self.DEBUG:
                            print("[Top-Left Point] Case 1: Strong Cue in Uniform-Sampling")
                        r_TL = r_l
                        c_TL = c_l
                    else:
                        if self.DEBUG:
                            print("[Top-Left Point] Case 2: No Strong Cue in Uniform-Sampling")
                        #r_TL, c_TL = self._findAdaptiveSamples(r_l, c_l, _type=1)
                        r_TL, c_TL, stat_TL = self._findAdaptiveSamples_v2(r_l, c_l, _type=1)
                        
                    '''
                    Top-Right Point
                    '''
                    if(self.ori_image[r_l,c_h]<self.cond_threshold):
                        if self.DEBUG:
                            print("[Top-Right Point] Case 1: Strong Cue in Uniform-Sampling")
                        r_TR = r_l
                        c_TR = c_h
                    else:
                        if self.DEBUG:
                            print("[Top-Right Point] Case 2: No Strong Cue in Uniform-Sampling")
                        #r_TR, c_TR = self._findAdaptiveSamples(r_l, c_h, _type=2)
                        r_TR, c_TR, stat_TR = self._findAdaptiveSamples_v2(r_l, c_h, _type=2)
                        
                    '''
                    Bottom-Right Point
                    '''
                    if(self.ori_image[r_h,c_h]<self.cond_threshold):
                        if self.DEBUG:
                            print("[Bottom-Right Point] Case 1: Strong Cue in Uniform-Sampling")
                        r_BR = r_h
                        c_BR = c_h
                    else:
                        if self.DEBUG:
                            print("[Bottom-Right Point] Case 2: No Strong Cue in Uniform-Sampling")
                        #r_BR, c_BR = self._findAdaptiveSamples(r_h, c_h, _type=3)
                        r_BR, c_BR, stat_BR = self._findAdaptiveSamples_v2(r_h, c_h, _type=3)
                    
                    '''
                    Bottom-Left Point
                    '''
                    if(self.ori_image[r_h,c_l]<self.cond_threshold):
                        if self.DEBUG:
                            print("[Bottom-Left Point] Case 1: Strong Cue in Uniform-Sampling")
                        r_BL = r_h
                        c_BL = c_l
                    else:
                        if self.DEBUG:
                            print("[Bottom-Left Point] Case 2: No Strong Cue in Uniform-Sampling")
                        #r_BL, c_BL = self._findAdaptiveSamples(r_h, c_l, _type=4)
                        r_BL, c_BL, stat_BL = self._findAdaptiveSamples_v2(r_h, c_l, _type=4)
                    
                    
                    if self.DEBUG:
                        print("Uniform:  ({},{}) ({},{}) ({},{}) ({},{})".format(r_l,c_l,r_l,c_h,r_h,c_h,r_h,c_l))
                        print("Adaptive: ({},{}) ({},{}) ({},{}) ({},{})".format(r_TL,c_TL,r_TR,c_TR,r_BR,c_BR,r_BL,c_BL))
        
                    p_TL = [r_TL,c_TL]
                    p_TR = [r_TR,c_TR]
                    p_BR = [r_BR,c_BR]
                    p_BL = [r_BL,c_BL]
                    
                    if self.DEBUG:
                        print("p_TL:{} p_TR:{} p_BR:{} p_BL:{} r:{} c:{}".format(p_TL, p_TR, p_BR, p_BL, r, c))
                    
                    stats = [stat_TL,stat_TR,stat_BR,stat_BL]
                    pixel = self._interpolate(p_TL, p_TR, p_BR, p_BL, r, c, amp, stats)
                    
                    self.resized_image[r][c] = pixel
                   
        else:
            raise NotImplementedError
            
        print('Done Resizing.')
        
    '''
    _sigmoid
    '''    
    def _sigmoid(self, x, offset=0, coef=0.05):
        return 1/(1+math.exp(-coef*(x-offset)))
        
    '''
    _amplify
    '''
    def _amplify(self, val_prev, val_new, param=0.75):
        val_prev = int(val_prev)
        val_new = int(val_new)
        if param==0:
            res = np.uint8(val_prev)
        else:
            res = np.uint8(max(0,val_new - abs(((val_prev-val_new)*param))))
        return res
    
    '''
    _interpolate
    '''
    def _interpolate(self, p_TL, p_TR, p_BR, p_BL, r, c, amplify, stats, EPS=1e-3):
        r_TL,c_TL = p_TL
        r_TR,c_TR = p_TR
        r_BR,c_BR = p_BR
        r_BL,c_BL = p_BL
        
        stat_TL, stat_TR, stat_BR, stat_BL = stats
        
        
        _target_c = self.w_ratio * c
        _target_r = self.h_ratio * r
                
        p_target = np.array((_target_r,_target_c))
        if self.DEBUG:
            print("Target Location: {}".format(p_target))
            
        d_TL = np.linalg.norm(p_target - np.array(p_TL)) + EPS
        d_TR = np.linalg.norm(p_target - np.array(p_TR)) + EPS
        d_BR = np.linalg.norm(p_target - np.array(p_BR)) + EPS
        d_BL = np.linalg.norm(p_target - np.array(p_BL)) + EPS
        
        if self.DEBUG:
            print("d_TL: {}".format(d_TL))
            print("d_TR: {}".format(d_TR))
            print("d_BR: {}".format(d_BR))
            print("d_BL: {}".format(d_BL))

        w_TL = 1/d_TL
        w_TR = 1/d_TR
        w_BR = 1/d_BR
        w_BL = 1/d_BL

        W = w_TL + w_TR + w_BR + w_BL

        norm_w_TL = w_TL/W
        norm_w_TR = w_TR/W
        norm_w_BR = w_BR/W
        norm_w_BL = w_BL/W
        
        '''u_pixel'''
        u_c_l, u_r_l = max(math.floor(_target_c),0), max(math.floor(_target_r),0)
        u_c_h, u_r_h = min(math.ceil(_target_c),self.w_old-1), min(math.ceil(_target_r),self.h_old-1)
        
        u_pixel_TL = self.ori_image[u_r_l, u_c_l]
        u_pixel_TR = self.ori_image[u_r_h, u_c_l]
        u_pixel_BR = self.ori_image[u_r_h, u_c_h]
        u_pixel_BL = self.ori_image[u_r_l, u_c_h]
        
        '''a_pixel'''
        pixel_TL = self.ori_image[r_TL,c_TL]
        pixel_TR = self.ori_image[r_TR,c_TR]
        pixel_BR = self.ori_image[r_BR,c_BR]
        pixel_BL = self.ori_image[r_BL,c_BL]
        if self.DEBUG:
            print("pixel_TL: {}".format(pixel_TL))
            print("pixel_TR: {}".format(pixel_TR))
            print("pixel_BR: {}".format(pixel_BR))
            print("pixel_BL: {}".format(pixel_BL))
        
        '''am_pixel'''
        if stat_TL['std'] >= self.std_th2:
            alpha_TL = 0.9 
        elif stat_TL['std'] <= self.std_th1:
            alpha_TL = 0.1
        else:
            alpha_TL = 0
            
        if stat_TR['std'] >= self.std_th2:
            alpha_TR = 0.9 
        elif stat_TR['std'] <= self.std_th1:
            alpha_TR = 0.1
        else:
            alpha_TR = 0
            
        if stat_BR['std'] >= self.std_th2:
            alpha_BR = 0.9 
        elif stat_BR['std'] <= self.std_th1:
            alpha_BR = 0.1
        else:
            alpha_BR = 0
            
        if stat_BL['std'] >= self.std_th2:
            alpha_BL = 0.9 
        elif stat_BL['std'] <= self.std_th1:
            alpha_BL = 0.1
        else:
            alpha_BL = 0

        self.alpha.append(alpha_TL)
        self.alpha.append(alpha_TR)
        self.alpha.append(alpha_BR)
        self.alpha.append(alpha_BL)
        self.alpha_std.append(stat_TL['std'])
        self.alpha_std.append(stat_TR['std'])
        self.alpha_std.append(stat_BR['std'])
        self.alpha_std.append(stat_BL['std'])
        am_pixel_TL = self._amplify(u_pixel_TL, pixel_TL, param = alpha_TL)
        am_pixel_TR = self._amplify(u_pixel_TR, pixel_TR, param = alpha_TR)
        am_pixel_BR = self._amplify(u_pixel_BR, pixel_BR, param = alpha_BR)
        am_pixel_BL = self._amplify(u_pixel_BL, pixel_BL, param = alpha_BL)

        if self.DEBUG:
            print("norm_ws: {},{},{},{}".format(norm_w_TL,norm_w_TR,norm_w_BR,norm_w_BL))
            print("sum(norm_ws): {}".format(norm_w_TL+norm_w_TR+norm_w_BR+norm_w_BL))

        pixel_weighted = np.round(pixel_TL*norm_w_TL + pixel_TR*norm_w_TR + pixel_BR*norm_w_BR + pixel_BL*norm_w_BL)
        if self.DEBUG:
            print("weighted pixel value: {}".format(pixel_weighted))
        assert 0<=pixel_weighted
        assert pixel_weighted<=255
        
        am_pixel_weighted = np.round(am_pixel_TL*norm_w_TL + am_pixel_TR*norm_w_TR + am_pixel_BR*norm_w_BR + am_pixel_BL*norm_w_BL)
        if self.DEBUG:
            print("amplified_weighted pixel value: {}".format(am_pixel_weighted))
        assert 0<=am_pixel_weighted
        assert am_pixel_weighted<=255
        
        if amplify:
            return am_pixel_weighted.astype(np.uint8)
        else:
            return pixel_weighted.astype(np.uint8)

    '''
    _findAdaptiveSamples_v2
    '''
    def _findAdaptiveSamples_v2(self, _r, _c, _type):
        stat={}
        scope_top    = None
        scope_bottom = None
        scope_left   = None
        scope_right  = None
        a_r, a_c = None, None

        scope_top    = max(0, _r-self.scope_scale)
        scope_bottom = min(self.h_old-1, _r+self.scope_scale)
        scope_left   = max(0, _c-self.scope_scale)
        scope_right  = min(self.w_old-1, _c+self.scope_scale)

        if self.DEBUG:
            print("Scope window [t,b,l,r]: [{},{},{},{}]".format(scope_top,scope_bottom,scope_left,scope_right))

        # Compute intensity stat within the window
        stat['avg'] = np.mean(self.ori_image[scope_top:scope_bottom+1, scope_left:scope_right+1])
        stat['std'] = np.std(self.ori_image[scope_top:scope_bottom+1, scope_left:scope_right+1])

        # A new set of sampling is started, so clear the buffer of sampling 4 points
        if(_type==1):
            self.buffer = []

        # Determine the window
        self.scope = self.map[scope_top:scope_bottom+1, scope_left:scope_right+1]
        # If no interesting entropy 
        if(np.sum(self.scope)==0):
            if self.DEBUG:
                print("\tsum(scope) is 0 (Follow uniformly sampled points)")
            #_r_scope, _c_scope = scope_top, scope_left
            return _r, _c, stat
        
        # mitigate bleed-through
        #elif stat['std']<self.ori_image_std and stat['avg']<self.ori_image_avg:
            #_r_scope, _c_scope = scope_top, scope_left
            return _r, _c, stat
        else:
            _r_scope_candidates, _c_scope_candidates = np.unravel_index(np.argsort(-self.scope.ravel()), self.scope.shape)

            for i in range(len(_r_scope_candidates)):            
                a_r, a_c = int(scope_top+_r_scope_candidates[i]), int(scope_left+_c_scope_candidates[i])

                # Check if newly sampled point is already sampled previously.
                # If so, skip the point and resample the pixel of having the next largest map value.
                if [a_r,a_c] in self.buffer:
                    continue
                else:
                    if self.DEBUG:
                        print("buffer:{} <- [{},{}]".format(self.buffer, a_r,a_c))

                    self.buffer.append([a_r,a_c])
                    break
            return a_r, a_c, stat

        
############
# argparse #
############
parser = argparse.ArgumentParser(description='Adaptively downscaling an image.')
parser.add_argument('--image_list', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./outputs/')
parser.add_argument('--output_prefix', type=str, default='')
parser.add_argument('--index_start', type=int, default=-1)
parser.add_argument('--index_end', type=int, default=-1)
parser.add_argument('--mode', type=str, default='uniform')
parser.add_argument('--option', type=str, default='log')
args = parser.parse_args()

IMAGE_LIST_FILE = args.image_list
OUTPUT_PREFIX   = args.output_prefix
INDEX_START     = args.index_start
INDEX_END       = args.index_end
OUTPUT_DIR      = args.output_dir
OUTPUT_CSV_NAME = OUTPUT_PREFIX + '_' + str(INDEX_START) + '_' + str(INDEX_END) + '.csv'
MODE            = args.mode
OPT             = args.option

print('Resizing image... {} + {}'.format(MODE,OPT))

# Prepare input stream
with open(IMAGE_LIST_FILE, 'r') as f:
    image_lists = f.read().split()
print("Total {} images are found.".format(len(image_lists)))
if INDEX_START==-1: INDEX_START=0
if INDEX_END==-1: INDEX_END=len(image_lists)    
print("Processing images from-to [{}-{}]".format(INDEX_START,INDEX_END))
image_lists = image_lists[INDEX_START:INDEX_END]

# Instantiate
resizer = ContentsAwareAdaptiveDownscaling()

for image_path in tqdm(image_lists):
    # Read image
    filename, file_extension = os.path.splitext(os.path.basename(image_path))
    image = cv2.imread(image_path,0)
    # Resize image
    #resizer.resize(image, size=(1280, 896), mode='uniform')
    #resizer.resize(image, size=(1280, 896), mode='adaptive', glcm='entropy', amp=False)    
    #resizer.resize(image, size=(1280, 896), mode='adaptive', glcm='entropy', amp=True)    
    #resizer.resize(image, size=(1280, 896), mode='adaptive', amp=False)
    resizer.resize(image, size=(1280, 896), mode=MODE, opt=OPT, amp=True)    
    
    # f2
    #resizer.resize(image, size=(640, 448), mode='uniform')
    #resizer.resize(image, size=(640, 448), mode='adaptive', amp=True)    
    
    # f4
    #resizer.resize(image, size=(320, 224), mode='uniform')
    #resizer.resize(image, size=(320, 224), mode='adaptive', amp=True)    
    
    # f8
    #resizer.resize(image, size=(160, 112), mode='uniform')
    #resizer.resize(image, size=(160, 112), mode='adaptive', amp=True)    

    # f16
    #resizer.resize(image, size=(80, 56), mode='uniform')
    #resizer.resize(image, size=(80, 56), mode='adaptive', amp=True)    
    
    # Save image
    OUTPUT_IMAGE_NAME = filename + '.png'
    cv2.imwrite(os.path.join(OUTPUT_DIR,OUTPUT_IMAGE_NAME), resizer.resized_image)
    
    # Log [image path, orignal shape, new shape]
    res = [image_path, resizer.h_old, resizer.w_old, resizer.h_new, resizer.w_new]
    with open(os.path.join(OUTPUT_DIR,OUTPUT_CSV_NAME), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(res)

print('Done.')
    
    