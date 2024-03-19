import numpy as np
import cv2

##======================== No additional imports allowed ====================================##

def photometric_stereo_singlechannel(I, L):
    #L is 3 x k
    #I is k x n
    G = np.linalg.inv(L @ L.T) @ L @ I
    # G is  3 x n 
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals

'''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N images, each a numpy float array of size H x W x 3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''
def photometric_stereo(images, lights):
    # Convert list of images to a 4D numpy array for easier manipulation
    images = np.stack(images, axis=-1)  # Now images is H x W x 3 x N

    H, W, C, N = images.shape

    # Initialize arrays for albedo and normals
    albedo = np.zeros((H, W, C))
    normals = np.zeros((H, W, 3))  # Average normals, so only one 3D vector per pixel

    for channel in range(C):
        # Reshape images to k x n, where k is number of images (lights) and n is number of pixels
        I = images[:, :, channel, :].reshape((H * W, N)).T  # I is now N x (H*W)

        # Compute albedo and normals for the current channel
        albedo_channel, normals_channel = photometric_stereo_singlechannel(I, lights)
        
        # Store the computed albedo
        albedo[:, :, channel] = albedo_channel.reshape(H, W)
        
        # Accumulate normals for averaging
        normals += normals_channel.T.reshape(H, W, 3)
    
    # Average the normals across all color channels
    normals /= C

    # Renormalize normals to ensure they are of unit length
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + (norm == 0).astype(float))  # Avoid division by zero

    return albedo, normals