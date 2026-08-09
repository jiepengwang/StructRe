import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
import glob, os
import cv2
from tqdm import tqdm
import numpy as np
import copy

import utils.utils_io as IOUtils

DIR_FILE = os.path.abspath(os.path.dirname(__file__))

def read_image(path, target_img_size = None, interpolation=cv2.INTER_LINEAR, color_space='BGR'):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if target_img_size is not None:
        img= resize_image(img, target_img_size, interpolation=interpolation)
    if color_space == 'RGB':
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img

def write_image(path, img, color_space = None, 
                    target_img_size = None, interpolation = cv2.INTER_LINEAR, check_path=True):
    '''If color space is defined, convert colors to RGB mode
    Args:
        target_img_size: resize image if defined
    
    '''
    if check_path:
        IOUtils.ensure_pdir_existence(path)
        
    if color_space == 'RGB':
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_img_size is not None:
        img = resize_image(img, target_size=target_img_size, interpolation=interpolation)
    cv2.imwrite(path, img)

def read_images(dir, target_img_size = None, interpolation=cv2.INTER_LINEAR, img_ext = '.png', use_rgb_mode = False, vec_stems = None):
    f'''Read images in directory with extrension {img_ext}
    Args:
        dir: directory of images
        target_img_size: if not none, resize read images to target size
        img_ext: defaut {img_ext}
        use_rgb_mode: convert brg to rgb if true
    Return:
        imgs: N*W*H
        img_stems
    '''
    read_img = lambda path : read_image(path) if not img_ext == '.npz' else np.load(path)['arr_0']

    vec_path = sorted(glob.glob(f"{dir}/**{img_ext}"))
    if vec_stems is not None:
        vec_path = []
        for stem_curr in vec_stems:
            vec_path.append(f'{dir}/{stem_curr}{img_ext}')
        vec_path = sorted(vec_path)

    rgbs = []
    img_stems = []
    for i in range(len(vec_path)):
        img = read_img(vec_path[i])
        if (target_img_size != None) and (target_img_size[0] != img.shape[1]):
            img= cv2.resize(img, target_img_size, interpolation=cv2.INTER_LINEAR)
        if use_rgb_mode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbs.append(img)

        _, stem, _ = IOUtils.get_path_components(vec_path[i])
        img_stems.append(stem)
    return np.array(rgbs), img_stems

def add_text_on_image(img, text, org = (20,50), fontScale=1, color = (0,0,255), thickness=2, lineType = cv2.LINE_AA):
    '''
    Args:
        color: bgr
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, fontScale, color, thickness=thickness, lineType = lineType)
    return img

def create_video_from_lis_images(path_video, lis_images, lis_tag_add=None, fps=8, ext= '.png', mode_input='path'):
    '''Merge dtu images to .avi video
    '''
    vec_path_imgs = lis_images # sorted(glob.glob(f'{dir_images}/**{ext}'))
    if mode_input == 'path':
        height, width, _ = cv2.imread(vec_path_imgs[0]).shape
    else:
        height, width, _ = lis_images[0].shape
    
    out_mode = cv2.VideoWriter_fourcc('M','J','P','G')
    # out_mode = cv2.VideoWriter_fourcc(*'DIVX')
    out_video = cv2.VideoWriter(path_video, out_mode , fps, (width, height))

    for i in tqdm(range(len(vec_path_imgs))):
        if mode_input == 'path':
            img = cv2.imread(vec_path_imgs[i])
            _, stem, _ = IOUtils.get_path_components(vec_path_imgs[i])
        else:
            img = lis_images[i]

        # add text
        # img = add_text_on_image(img, stem.split('_')[0][2:], org = (20,30))
        if lis_tag_add is None:
            content_add = f'{i:03d}'
        else:
            content_add = lis_tag_add[i]
        img = add_text_on_image(img, content_add, org = (20,30))

        out_video.write(img)
    cv2.destroyAllWindows()
    out_video.release()

def resize_image(img, target_size, interpolation=cv2.INTER_LINEAR):
    '''Resize image to target size
    Args:
        target_size: (W,H)
    Return img_resize
    
    '''
    if img is None:
        # print(f'Image is None')
        return  np.ones((384, 384, 3))*0
    W,H = target_size[0], target_size[1]
    if img.shape[0] != H or img.shape[1] != W:
        img_resize = cv2.resize(img, (W,H), interpolation=interpolation)
        return img_resize
    else:
        return img

def draw_histo_arr(path_fig, arr_data, n_his, thres_clip=None):
    msg_stat = f'Total elem: {arr_data.size}; shape: {arr_data.shape}. \nData (mean_abs, mean, min, max): {np.absolute(arr_data).mean():.4f}, {arr_data.mean():.04f},       {arr_data.min():.04f}, {arr_data.max():.04f}'
    print(msg_stat)
    
    if thres_clip is not None:
        arr_data = arr_data.clip(thres_clip)
    plt.hist(arr_data.reshape(-1), n_his)
    # x_ticks = np.arange(round(arr_data.min()) , round(arr_data.max()), n_his//2)
    # plt.xticks(x_ticks)
    plt.title(msg_stat)
    plt.savefig(path_fig)   
    plt.close()

def draw_two_curve(path_fig, arr_data1, arr_data2, label1, label2):
    
    idx = np.arange(len(arr_data1))
    plt.plot(idx, arr_data1, 'b', label=label1)
    plt.plot(idx, arr_data2, 'r', label=label2)
    
    plt.legend(loc='upper right')
    plt.savefig(path_fig)   
    plt.close()
            
def convert_gray_to_cmap(img_gray, map_mode = 'jet', revert = True, vmax = None):
    '''Visualize point distances with 'hot_r' color map
    Args:
        cloud_source
        dists_to_target
    Return:
        cloud_visual: use color map, max
    '''
    img_gray = copy.deepcopy(img_gray)
    shape = img_gray.shape

    cmap = plt.get_cmap(map_mode)
    if vmax is not None:
        img_gray = img_gray / vmax
    else:
        img_gray = img_gray / (np.max(img_gray)+1e-6)
    if revert:
        img_gray = 1- img_gray      
    colors = cmap(img_gray.reshape(-1))[:, :3]

    # visualization
    colors = colors.reshape(shape+tuple([3]))*255
    return colors

def add_black_border(img):
    lis_idx_add = [0, 1, 2,  -3, -2, -1]
    img[lis_idx_add, :] = 0
    img[:, lis_idx_add] = 0
    return img

def unify_image_width(lis_img_all):
    """Unify the width of images in lis_img_all
    """
    
    width_max = 0
    for i in range(len(lis_img_all)):
        img_merge = lis_img_all[i]
        if img_merge.shape[1]>=width_max:
            width_max = img_merge.shape[1]           

    if width_max>0:
        lis_img_all_align = []
        for i in range(len(lis_img_all)):
            img_new = np.ones((lis_img_all[0].shape[0], width_max, 3), dtype=np.uint8)*255
            img_new[:, :lis_img_all[i].shape[1]] = lis_img_all[i]
            lis_img_all_align.append(img_new)      
    else:
        lis_img_all_align = lis_img_all
    return lis_img_all_align

def unify_image_to_grids(lis_img_all_align):
    n_col = 4; n_row = 3; interval=20; n_grid_max = 100
    lis_img_all_align2 = []
    for i in range(len(lis_img_all_align)):
        image_i = lis_img_all_align[i]
        h, w = image_i.shape[:2]
        
        lis_grid = []
        for j in range(n_grid_max):
            w_start = j*(h+interval)
            w_end = j*(h+interval)+ h
            if w_end <= w:
                grid_curr = image_i[:, w_start:w_end]
                lis_grid.append(grid_curr)
        
        # reorganize grid
        n_grid_curr = len(lis_grid)

        image_width = (h+interval)*n_col-interval
        image_height = (h+interval)*n_row-interval
        image_new = np.ones((image_height, image_width, 3), dtype=np.uint8)*255
        
        # fill grid
        for i_row in range(n_row):
            for j_col in range(n_col):
                idx_grid = i_row*n_col+j_col
                if idx_grid >= n_grid_curr:
                    break
                image_new[i_row*(h+interval):i_row*(h+interval)+h, j_col*(h+interval):j_col*(h+interval)+h] = lis_grid[idx_grid]
        
        lis_img_all_align2.append(image_new) 
    return lis_img_all_align2

def write_image_lis(path_img_cat, lis_imgs, use_cmap = False, interval_img = 20, cat_mode = 'horizontal', color_space = 'BGR', 
                        add_black_border=False, size_image=None, check_side_max=False, tag_text=None, reso_downsample=-1):
    '''Concatenate an image list to a single image and save it to the target path
    Args:
        cat_mode: horizonal/vertical
    '''
    if path_img_cat is not None:
        IOUtils.ensure_pdir_existence(path_img_cat)
    
    img_cat = []

    for i in range(len(lis_imgs)):
        img = lis_imgs[i]

        if check_side_max:
            if i==0:
                H_Max, W_Max = img.shape[:2]
            else:
                H, W = img.shape[:2]
                H_Max = H if H>H_Max else H_Max
                W_Max = W if W>W_Max else W_Max

    
    for i in range(len(lis_imgs)):
        img = lis_imgs[i]
        H, W = img.shape[:2]
        if img.ndim == 3 and img.shape[-1]==4:
            img = img[:, :, :3]

        # if cat_mode == 'horizontal':
        if check_side_max:
            if H<H_Max:
                img_expand = np.ones((H_Max, W, 3))*255
                img_expand[:H, :W] = img
                img= img_expand

                H=H_Max
        # elif cat_mode == 'vertical':
            if W<W_Max:
                img_expand = np.ones((H, W_Max, 3))*255
                img_expand[:H, :W] = img
                img= img_expand

                W=W_Max
        
        if size_image is not None:
            img = resize_image(img, size_image)
        
        if use_cmap:
            img = convert_gray_to_cmap(img) if img.ndim==2 else img
        else:
            img = np.stack([img]*3, axis=-1) if img.ndim==2 else img
        
        if img.max() <= 1.0:
            img *= 255
        
        if add_black_border:
            lis_idx_add = [0,1,2, -3, -2,-1]
            img[lis_idx_add, :] = 0
            img[:, lis_idx_add] = 0
        
        if tag_text is not None:
            img = add_text_on_image(img, f'{tag_text}_{i:02d}', color=(0, 0,255))

        img_cat.append(img)
        if cat_mode == 'horizontal':
            img_cat.append(255 * np.ones((H, interval_img, 3)).astype('uint8'))
        elif cat_mode == 'vertical':
            img_cat.append(255 * np.ones((interval_img, W, 3)).astype('uint8'))
    
    if cat_mode == 'horizontal':
        img_cat = np.concatenate(img_cat[:-1], axis=1)
    elif cat_mode == 'vertical':
        img_cat = np.concatenate(img_cat[:-1], axis=0)
    else:
        raise ValueError(
            f"Unsupported image concatenation mode {cat_mode!r}; "
            "expected 'horizontal' or 'vertical'"
        )
    if color_space == 'RGB':
        img_cat = cv2.cvtColor(img_cat.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    if reso_downsample > 0:
        img_cat = cv2.resize(img_cat, (img_cat.shape[1]//reso_downsample, img_cat.shape[0]//reso_downsample), interpolation=cv2.INTER_LINEAR)

    if path_img_cat:
        cv2.imwrite(path_img_cat, img_cat)
    return img_cat

def merge_image_parent_lis_child(img_parent, lis_img_child):
    img_cat_child = write_image_lis(None, lis_img_child, cat_mode = 'horizontal')
    img_parent_expand = np.ones_like(img_cat_child)*255
    h, w = img_parent.shape[:2]
    img_parent_expand[:h, :w] = img_parent
    img_parent_expand = img_parent_expand[:h]
    img_parent_expand = add_black_border(img_parent_expand)

    img_cat_pc = write_image_lis(None, [img_parent_expand, img_cat_child], cat_mode = 'vertical', interval_img = 20)
    return img_cat_pc
