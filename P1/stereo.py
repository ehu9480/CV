import numpy as np
#==============No additional imports allowed ================================#
'''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor
    
    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
def get_ncc_descriptors(img, patchsize):
    height, width, channels = img.shape
    normalized = np.zeros((height, width, channels * patchsize**2))
    
    for i in range(height):
        for j in range(width):
            patch = img[max(0, i-patchsize//2):min(height, i+patchsize//2+1),
                        max(0, j-patchsize//2):min(width, j+patchsize//2+1), :]
            # check if the patch is out of boundary for given center pixel (i,j)
            if (i - patch.size/2) < 0 or (i + patch.size/2) >= height or (j - patch.size/2) < 0 or (j + patch.size/2) >= width:
                normalized[i, j] = 0
            else:
                patch_mean = np.mean(patch, axis=(0, 1))
                patch_subtracted = patch - patch_mean
                patch_flattened = patch_subtracted.reshape(-1)
                patch_norm = np.linalg.norm(patch_flattened)
                
                if patch_norm < 1e-6:
                    normalized[i, j] = 0
                else:
                    normalized[i, j] = patch_flattened / patch_norm
    
    return normalized

'''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    ncc_vol = np.zeros((dmax, img_right.shape[0], img_right.shape[1]))
    right_descriptors = get_ncc_descriptors(img_right, patchsize)
    left_descriptors = get_ncc_descriptors(img_left, patchsize)

    for d in range(dmax):
        for i in range(img_right.shape[0]):
            for j in range(img_right.shape[1]):
                if j + d < img_right.shape[1]:
                    ncc_vol[d, i, j] = np.dot(right_descriptors[i, j], left_descriptors[i, j + d])
                else:
                    ncc_vol[d, i, j] = 0

    return ncc_vol

'''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
def get_disparity(ncc_vol):
    disparity = np.argmax(ncc_vol, axis=0)
    return disparity





    
