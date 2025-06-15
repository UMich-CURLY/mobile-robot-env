import numpy as np
def get_distance(depth_image,hfov=72,p = -50):
    cw = np.arctan(hfov/2/180*np.pi)
    ch = cw/640*480
    yv, xv = np.meshgrid(np.linspace(-ch,ch,480),np.linspace(-cw,cw,640), indexing='ij')
    px,py = xv*depth_image,yv*depth_image
    # print(np.max(px))
    # print(np.max(py))

    pcd  = np.stack([depth_image,-px,py],axis=-1) # robot frame pointcloud, H x W x 3 (xyz)
    #generalized mean
    distances = np.linalg.norm(pcd,axis=2)[:220,:] #depth image to distance image.\
    return (np.sum(distances**p)/640/480)**(1/p)#np.mean(distances)
