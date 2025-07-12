import numpy as np
import cv2
def get_distance(depth_image,hfov=54.7,p=-30):
    # resize depth image to 320x240
    # depth_image = cv2.resize(depth_image, (320, 240))
    height, width = depth_image.shape

    cw = np.arctan(hfov/2/180*np.pi)
    ch = cw/width*height
    yv, xv = np.meshgrid(np.linspace(-ch,ch,height),np.linspace(-cw,cw,width), indexing='ij')
    px,py = xv*depth_image,yv*depth_image
    # print(np.max(px))
    # print(np.max(py))

    pcd  = np.stack([depth_image,-px,py],axis=-1) # robot frame pointcloud, H x W x 3 (xyz)
    #generalized mean
    distances = np.linalg.norm(pcd,axis=2)[:int(0.6*height),:] #depth image to distance image.\
    distances = distances[distances>0.1]
    return (np.sum(distances**p)/width/height)**(1/p)#np.mean(distances)
