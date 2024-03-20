import numpy as np

##======================== No additional imports allowed ====================================##

def photometric_stereo_singlechannel(I, L):
    G = np.linalg.inv(L @ L.T) @ L @ I
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals

def photometric_stereo(images, lights):
    images = np.stack(images, axis=-1)

    H, W, C, N = images.shape

    albedo = np.zeros((H, W, C))
    normals = np.zeros((H, W, 3)) 

    for channel in range(C):
        I = images[:, :, channel, :].reshape((H * W, N)).T
        albedo_channel, normals_channel = photometric_stereo_singlechannel(I, lights)
        albedo[:, :, channel] = albedo_channel.reshape(H, W)
        normals += normals_channel.T.reshape(H, W, 3)

    normals /= C
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / (norm + (norm == 0).astype(float))
    return albedo, normals
