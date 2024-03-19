import numpy as np
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
    H, W, _ = images[0].shape
    N = len(images)
    k = lights.shape[1]
    
    albedo = np.zeros((H, W, 3))
    normals = np.zeros((H, W, 3))
    
    for i in range(N):
        I = images[i].reshape(-1, 3).T
        L = lights[:, i].reshape(3, 1)
        
        albedo_channel, normals_channel = photometric_stereo_singlechannel(I, L)
        
        albedo += albedo_channel.reshape(H, W, 1)
        normals += normals_channel.reshape(H, W, 1)
    
    albedo /= N
    normals /= N
    
    # Renormalize normals
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)
    
    return albedo, normals



