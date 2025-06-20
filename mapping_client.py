import pygame
from PIL import Image
import numpy as np
import socket
import pickle
import time
from collections import deque
import math
import pygame.colordict

from utils.protocol import *
from argparse import ArgumentParser
from utils.planner import fit_smoothing_spline
HFOV = 54.7
#preprogrammed waypoints to execute by pressing enter.
# WAYPOINTS = np.array([
# [0,0],[0.5,-0.3],[1,0.2],[1.5,0],[2,-0.2],[2.5,0]
# ])
WAYPOINTS = np.array([
 [ 0.         , 0.        ],
 [ 0.64484999 , 0.00898335],
 [ 0.98256157 , 0.00693384],
 [ 1.1899422  , 0.14765533],
 [ 1.34841325 , 0.50032   ],
 [ 1.60909961 , 0.71981093],
 [ 1.96884275 , 0.81072732],
 [ 2.21724862 , 0.66540045],
 [ 2.48945878 , 0.48726669],
 [ 2.60769912 , 0.23764456],
 [ 2.54893836 ,-0.07430095],
 [ 2.34404152 ,-0.36827063],
 [ 2.11184586 ,-0.39968225],
 [ 1.85050507 ,-0.32115638],
 [ 1.46089797 ,-0.2089194 ],
 [ 1.26725344 ,-0.35803746]])
WAYPOINTS = np.array([

[0.,0.],[0.1,0],[0.5,0],[1,0]
])


MAGNIFICATION_OPTIONS = [1,2,4,6,8]
magnification_choice = 3

parser = ArgumentParser()
parser.description = "Welcome to the robot client for the SG VLN project"
parser.add_argument("--host",type=str,default='localhost',help="the host name of the remote robot server")
args = parser.parse_args()
pygame.init()
BORDER = 30
ROBOT_VIS_CENTER = np.array([BORDER*2+640+320,240+BORDER])
screen_width, screen_height = 640*2+BORDER*3, 720+20
window = pygame.display.set_mode((screen_width, screen_height),pygame.RESIZABLE)
pygame.display.set_caption("SG-VLN WEBSOCKET CLIENT")
view_rgb = True

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()
def draw_compass_arrow(
    surface: pygame.Surface,
    x: int,
    y: int,
    yaw_radians: float,
    length: int = 20,
    width: int = 20,
    tail_length: int = None,
    tail_width: int = None,
    color_head: tuple = (255, 0, 0),  # Red for North
):
    """
    Draws a compass arrow on a Pygame surface.

    Args:
        surface (pygame.Surface): The surface to draw on (e.g., screen).
        x (int): X-coordinate of the arrow's pivot point (center of its base).
        y (int): Y-coordinate of the arrow's pivot point (center of its base).
        yaw_radians (float): The rotation angle in radians.
                             0 radians points up (North). Positive rotates clockwise.
        length (int): Length of the arrow's head part from pivot to tip.
        width (int): Maximum width of the arrow's head part at its base.
        tail_length (int, optional): Length of the arrow's tail part from pivot.
                                     Defaults to length / 2 if None.
        tail_width (int, optional): Width of the arrow's tail part at its base.
                                    Defaults to width if None.
        color_head (tuple): RGB color for the arrow's head.
        color_tail (tuple): RGB color for the arrow's tail.
    """
    if tail_length is None:
        tail_length = length // 2
    if tail_width is None:
        tail_width = width

    # Calculate the maximum dimension needed for the temporary surface to contain the
    # rotated arrow without clipping. A simple way is to use the diagonal of the
    # bounding box, or just a large enough multiple of the longest part.
    max_dim = max(length, width, tail_length, tail_width) * 2

    # Create a temporary transparent surface for drawing the arrow.
    # This is crucial for rotating the arrow without a black background.
    temp_surface = pygame.Surface((max_dim, max_dim), pygame.SRCALPHA)

    # Calculate the center of the temporary surface. This will be our drawing pivot.
    cx, cy = max_dim // 2, max_dim // 2

    # Define points for the arrow's head (relative to the pivot cx, cy)
    # Arrow points initially straight up (North)
    head_points = [
        (cx, cy - length/2),        # Tip of the arrow
        (cx - width // 2, cy+length/2),  
          (cx,cy),  # Bottom-left of the head base
        (cx + width // 2, cy+length/2)     # Bottom-right of the head base
    ]

    # Draw the head and tail on the temporary surface
    pygame.draw.polygon(temp_surface, color_head, head_points,2)
    rotated_arrow = pygame.transform.rotate(temp_surface, math.degrees(yaw_radians-np.pi/2))
    arrow_rect = rotated_arrow.get_rect(center=(x, y))

    # Blit the rotated arrow onto the main surface
    surface.blit(rotated_arrow, arrow_rect)
clock = pygame.time.Clock()

from utils.socket_client import request_sensor_data,send_action_message,request_planner_state

run = True
data = None

lastx = 0 
lasty = 0

points = deque(np.zeros((0,2),dtype = float),maxlen=3000)
vx,vy,omg = 0,0,0

init_T = None
curr_T = None

from scipy.spatial.transform import Rotation
from copy import deepcopy



waypointmsg = WaypointMessage()
translations = None

pcd_memory = deque(maxlen=100)
import matplotlib.cm as cm   
def depth_to_pil_rgb(depth_array, 
                     cmap_name='viridis', 
                     min_depth=None, 
                     max_depth=None, 
                     nan_color=(0, 0, 0), # RGB tuple for NaN/Inf values (0-255)
                     valid_mask=None):
    """
    Converts a depth image (HxW float array) to a PIL RGB image.

    Args:
        depth_array (np.ndarray): HxW NumPy array of floats representing depth.
        cmap_name (str): Name of the Matplotlib colormap to use (e.g., 'viridis', 'jet', 'gray').
        min_depth (float, optional): Minimum depth value for normalization. 
                                     If None, it's taken from the array's min.
        max_depth (float, optional): Maximum depth value for normalization.
                                     If None, it's taken from the array's max.
        nan_color (tuple, optional): RGB tuple (0-255) to use for NaN or Inf values in the depth_array.
                                     If None, Matplotlib's default bad color handling is used (often transparent or first color).
        valid_mask (np.ndarray, optional): HxW boolean NumPy array. If provided, only pixels where
                                           mask is True are considered for min/max calculation and colormapping.
                                           Pixels where mask is False will be colored with `nan_color`.

    Returns:
        PIL.Image.Image: An RGB PIL Image.
    """
    if not isinstance(depth_array, np.ndarray) or depth_array.ndim != 2:
        raise ValueError("depth_array must be a 2D NumPy array.")

    # Make a copy to avoid modifying the original array
    processed_depth = depth_array.copy()

    # Handle explicit valid_mask
    invalid_mask_explicit = None
    if valid_mask is not None:
        if valid_mask.shape != processed_depth.shape or valid_mask.dtype != bool:
            raise ValueError("valid_mask must be a boolean array with the same shape as depth_array.")
        invalid_mask_explicit = ~valid_mask
        processed_depth[invalid_mask_explicit] = np.nan # Mark these as NaN for consistent handling

    # Identify NaN and Inf values
    is_invalid = np.isnan(processed_depth) | np.isinf(processed_depth)
    
    # Determine normalization range, considering only valid (non-NaN, non-Inf) pixels
    valid_pixels = processed_depth[~is_invalid]

    if min_depth is None:
        min_val = np.min(valid_pixels) if valid_pixels.size > 0 else 0
    else:
        min_val = min_depth

    if max_depth is None:
        max_val = np.max(valid_pixels) if valid_pixels.size > 0 else 1
    else:
        max_val = max_depth

    # Handle edge case: all values are the same or no valid pixels
    if min_val >= max_val:
        # If all values are the same (or range is zero), map them to the lower end of cmap
        # NaNs/Infs will still be handled separately.
        normalized_depth = np.zeros_like(processed_depth, dtype=np.float32)
        # Ensure valid pixels that are equal to min_val/max_val get 0
        normalized_depth[~is_invalid] = 0 
    else:
        # Clip and normalize valid values to [0, 1]
        # First, ensure that values outside min_val/max_val are clipped
        # We need to handle NaNs carefully here. `np.clip` propagates NaNs.
        clipped_depth = np.clip(processed_depth, min_val, max_val)
        
        # Normalize
        normalized_depth = (clipped_depth - min_val) / (max_val - min_val)
    
    # Set NaNs in normalized_depth to a value that cmap will handle (e.g., 0 or nan)
    # This ensures they are processed by cmap, potentially using cmap.set_bad()
    normalized_depth[is_invalid] = np.nan 

    # Get the colormap
    try:
        cmap = cm.get_cmap(cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis'.")
        cmap = cm.get_cmap('viridis')

    # Apply the colormap. Matplotlib colormaps return RGBA values in [0, 1].
    # NaNs in normalized_depth will be mapped by cmap's bad_color logic
    # if cmap.set_bad() was used, or to a default (often transparent or first color).
    colored_array_rgba = cmap(normalized_depth) # Output is HxWx4, float in [0,1]

    # Convert to 8-bit RGB [0, 255]
    colored_array_rgb = (colored_array_rgba[:, :, :3] * 255).astype(np.uint8)

    # If a specific nan_color is provided, apply it to the original NaN/Inf locations
    if nan_color is not None:
        if not (isinstance(nan_color, tuple) and len(nan_color) == 3 and 
                all(isinstance(c, int) and 0 <= c <= 255 for c in nan_color)):
            raise ValueError("nan_color must be an RGB tuple of integers in [0, 255].")
        colored_array_rgb[is_invalid] = nan_color

    # Create PIL Image
    pil_image = Image.fromarray(colored_array_rgb, 'RGB')

    return pil_image
while run:
    if translations is None:
        send_action_message(VelMessage(vx,vy,omg), host=args.host)

    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            translations = None

            # print(Rotation.from_matrix(curr_T[:3,:3]).as_quat())
            if event.key == pygame.K_w:
                vx = 0.5
            if event.key == pygame.K_s:
                vx = -0.5
            if event.key == pygame.K_a:               
                omg = 0.5
            if event.key == pygame.K_d:
                omg = -0.5
            if event.key == pygame.K_q:
                vy = 0.5
            if event.key == pygame.K_e:
                vy = -0.5
            if event.key == pygame.K_SPACE:
                vx,vy,omg = vx*2,vy*2,omg*2
            if event.key == pygame.K_BACKSPACE:
                if(WAYPOINTS.shape[0]>1):
                    WAYPOINTS = WAYPOINTS[:-1]
                if(WAYPOINTS.shape[0]>1):
                    translations = np.hstack((WAYPOINTS,np.ones((len(WAYPOINTS),1))*0.2,np.ones((len(WAYPOINTS),1)))) @  curr_T.T# @ np.linalg.inv(init_T).T 

            if event.key == pygame.K_DELETE:
                WAYPOINTS = WAYPOINTS[:1]
            if event.key == pygame.K_RETURN:
                print("executing preprgrammed trajectory")
                if(WAYPOINTS.shape[0]>1):
                    print(f"Waypoint number: {WAYPOINTS.shape}")
                    translations = np.hstack((WAYPOINTS,np.ones((len(WAYPOINTS),1))*0.2,np.ones((len(WAYPOINTS),1)))) @  curr_T.T# @ np.linalg.inv(init_T).T 
                    
                    waypointmsg.x = translations[:,0].tolist()
                    waypointmsg.y = translations[:,1].tolist() #invert it because z is positive right, but y is positive left.

                    send_action_message(waypointmsg, host=args.host)
                else:
                    print("not enough points, skipping")
                continue
            if event.key == pygame.K_m:
                magnification_choice+=1
                magnification_choice%=len(MAGNIFICATION_OPTIONS)
            if event.key == pygame.K_n:
                magnification_choice-=1
                magnification_choice%=len(MAGNIFICATION_OPTIONS)
            if event.key == pygame.K_i:
                view_rgb = not view_rgb

            if event.key == pygame.K_o:
                print(WAYPOINTS)
            
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                vx = 0
            if event.key == pygame.K_s:
                vx = 0
            if event.key == pygame.K_a:               
                omg = 0
            if event.key == pygame.K_d:
                omg = 0
            if event.key == pygame.K_q:
                vy = 0
            if event.key == pygame.K_e:
                vy = 0
            if event.key == pygame.K_SPACE:
                vx,vy,omg = vx/2,vy/2,omg/2
            
            send_action_message(VelMessage(vx,vy,omg),args.host)
        
        if event.type ==  pygame.MOUSEBUTTONDOWN:
            mx,my = (np.array(pygame.mouse.get_pos())-ROBOT_VIS_CENTER*window.get_width()/screen_width-np.array([window.get_rect().x,window.get_rect().y]))*np.array([1,-1])/scale/window.get_width()*screen_width
            click_pos= np.linalg.inv(curr_T) @ np.array([mx+curr_T[0,3],my+curr_T[1,3],0.2,1])
            WAYPOINTS = np.vstack((WAYPOINTS,click_pos[:2].reshape((1,2))))
            translations = np.hstack((WAYPOINTS,np.ones((len(WAYPOINTS),1))*0.2,np.ones((len(WAYPOINTS),1)))) @  curr_T.T# @ np.linalg.inv(init_T).T 
            # wps = np.vstack((points[-1]-init_pos,points[-1]+np.array([x,y])*10-init_pos))

            # print(wps)
            # translations = np.hstack((wps,np.ones((len(wps),1))*0.2)) @ init_rot

            # waypoints = WaypointMessage()
            # waypoints.x = translations[:,0]
            # waypoints.z = translations[:,1]
            # send_action_message(waypoints,args.host)


    try:
        start_ts = time.time()
        data = request_sensor_data(args.host)
        end_ts = time.time()
        # print(f"[Request] Time: {end_ts - start_ts:.3f}s")
        my_dict = request_planner_state(args.host)
        formatted_dict = {key: f"{value:+.2f}" for key, value in my_dict.items()}
        if translations is not None:
            planner_message = "[PLANNER] "

            for key,value in formatted_dict.items():
                planner_message+=key
                planner_message+=": "
                planner_message+=value
                planner_message+=" | "
        else:
            collision = my_dict.get("collision",None)
            planner_message= f"[TELEOP] x: {vx:+.2f} y: {vy:+.2f} z: {omg:+.2f} | proximity warning: {collision}"
        # decimal_places = 3
        # rounded_dict = {k: round(v, decimal_places) if isinstance(v, float) else v for k, v in my_dict.items()}
        # print(rounded_dict)

    except socket.timeout:
            print(f"Socket timeout during operation with {SERVER_HOST}:{SERVER_PORT}")
    except socket.error as e:
        print(f"Socket error: {e}")
    except pickle.UnpicklingError as e:
        print(f"Pickle error: {e}. Received data might be corrupt or not a pickle.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if 'client_socket' in locals():
            client_socket.close()

    if data and data.get("success", False):
        screen = pygame.Surface((screen_width, screen_height))

        latency_ms = int((time.time_ns() - data.get('timestamp_server_ns',-1000000)) / 1000000)

        rgb_image = data.get("rgb_image")
        depth_image = data.get("depth_image").astype(float)/1000.0
        # print(np.max(depth_image))
        # convert depth image to distance image
        cw = np.arctan(HFOV/2/180*np.pi)
        ch = cw/640*480
        yv, xv = np.meshgrid(np.linspace(-ch,ch,480),np.linspace(-cw,cw,640), indexing='ij')
        px,py = xv*depth_image,yv*depth_image
        # print(np.max(px))
        # print(np.max(py))

        pcd  = np.stack([depth_image,-px,py],axis=-1)
        #generalized mean
        power = -50
        distances = np.linalg.norm(pcd,axis=2)[:300,:] #depth image to distance image.\
        distances = distances[distances>0.1]
        mean_distance = (np.sum(distances**power)/640/480)**(1/power)#np.mean(distances)
        end_ts = time.time()


        # print(f"mean distance {mean_distance}")
        # print(f"weighted distance: {}")
        # print(f"min distance {np.min(distances[depth_image.reshape((-1))>0.05])}")

        pose = data.get("pose")
        p = pose['pose']['position']
        o = pose['pose']['orientation']

        curr_T = np.eye(4)
        curr_T[:3,:3] = Rotation.from_quat([o['x'],o['y'],o['z'],o['w']]).as_matrix()
        curr_T[:3,3] = np.array([p['x'],p['y'],p['z']])
        curr_T = deepcopy(curr_T)
        hmax = 280
        hmin = 200
        random_indices = np.random.choice((hmax-hmin)*640, size=100, replace=False)
        sampled_pcd = pcd[hmin:hmax,:].reshape((-1,3))[random_indices]
        rotated_pcd = sampled_pcd @ curr_T[:3,:3].T

        transformed_pcd = np.hstack((sampled_pcd,np.ones(sampled_pcd[:,:1].shape))) @ curr_T.T
        pcd_memory.append(transformed_pcd)

        curr_yaw = Rotation.from_matrix(curr_T[:3,:3]).as_euler('zyx')[0]
        if init_T is None:
            init_T = deepcopy(curr_T)
            print(init_T)

        points.append([p['x'],p['y']])

        p = np.array(points)*np.array([[1,-1]])

        server_timestamp_ns = data.get("timestamp_server_ns")
        screen.fill(0)


        magnification_scale = MAGNIFICATION_OPTIONS[magnification_choice]
        scale = 10*magnification_scale
        n_line_x = 32//magnification_scale
        n_line_y = 24//magnification_scale

        # for point in rotated_pcd:
        #     # screen.set_at((point[:2]*scale*np.array([1,-1])).astype(int)+ROBOT_VIS_CENTER,pygame.Color('purple'))
        #     r,g,b = 255,point[2]*50+100,-point[2]*50+100
        #     r-= abs(point[2])*200
        #     g-= abs(point[2])*200
        #     b-= abs(point[2])*200
        #     pygame.draw.circle(screen,pygame.Color(int(np.clip(r,0.1,255)),int(np.clip(g,0.1,255)),int(np.clip(b,0.1,255))),(point[:2]*scale*np.array([1,-1])).astype(int)+ROBOT_VIS_CENTER,4,1)
        for transformed_pcd in pcd_memory:
            aligned_pcd = transformed_pcd @ np.linalg.inv(curr_T).T[:,:3] @ curr_T[:3,:3].T
            for point in aligned_pcd:
                r,g,b = 255,point[2]*50+100,-point[2]*50+100
                r-= abs(point[2])*200
                g-= abs(point[2])*200
                b-= abs(point[2])*200
                pygame.draw.circle(screen,pygame.Color(int(np.clip(r,0.1,255)),int(np.clip(g,0.1,255)),int(np.clip(b,0.1,255))),(point[:2]*scale*np.array([1,-1])).astype(int)+ROBOT_VIS_CENTER,4,1)
   
        for x in np.linspace(-scale*n_line_x,scale*n_line_x,2*n_line_x):
            pygame.draw.line(screen, (30*magnification_scale,30*magnification_scale,30*magnification_scale),ROBOT_VIS_CENTER+np.array([x,-scale*n_line_y]), ROBOT_VIS_CENTER+np.array([x,scale*n_line_y])) 
        for y in np.linspace(-scale*n_line_y,scale*n_line_y,2*n_line_y):
            pygame.draw.line(screen, (30*magnification_scale,30*magnification_scale,30*magnification_scale),ROBOT_VIS_CENTER+np.array([-scale*n_line_x,y]), ROBOT_VIS_CENTER+np.array([scale*n_line_x,y])) 

        if len(points)>1:
            for i in range(1,len(points)):
                pygame.draw.line(screen, pygame.Color('yellow'),(p[i-1]-p[-1])*scale+ROBOT_VIS_CENTER, (p[i]-p[-1])*scale+ROBOT_VIS_CENTER,2) 

        if translations is not None:

            spline,_,_ = fit_smoothing_spline(translations[:,:2],n=100)
            for i in range(1,len(spline)):
                pygame.draw.line(screen, pygame.Color('green'),(spline[i-1,:2]-curr_T[:2,3])*np.array([1,-1])*scale+ROBOT_VIS_CENTER, (spline[i,:2]-curr_T[:2,3])*np.array([1,-1])*scale+ROBOT_VIS_CENTER,1) 
        if view_rgb:
            pygameSurface = pilImageToSurface(Image.fromarray(rgb_image,mode='RGB'))
        else:
            pygameSurface = pilImageToSurface(depth_to_pil_rgb(depth_image))


        screen.blit(pygameSurface, (BORDER,BORDER))
        font = pygame.font.SysFont('Courier', 25)#pygame.font.Font('freesansbold.ttf', 32)
        
        mean_distance = np.clip(mean_distance,0,5)
        # Create a Rect object 
        r = mean_distance*scale
        rect = pygame.Rect(0, 0, r*2,r*2)
        rect.center = ROBOT_VIS_CENTER
        pygame.draw.arc(screen,(255-mean_distance*50,mean_distance*50,0),rect,curr_yaw-0.7,curr_yaw+0.7,int(8-mean_distance**2))


        draw_compass_arrow(screen,ROBOT_VIS_CENTER[0],ROBOT_VIS_CENTER[1],curr_yaw)

        distance_text = font.render(f"[INFO] mean distance: {mean_distance:.2f}m | map magnification: {magnification_scale}X | fps: {clock.get_fps():.1f} | E2E latency: {latency_ms:04} ms",True,(255,255,255))
        screen.blit(distance_text,(BORDER,480+BORDER*2))
        # create a text surface object,
        # on which text is drawn on it.
        green = (0, 255, 0)
        blue = (0, 0, 128)
        path_text = font.render(planner_message, True, green)
        screen.blit(path_text,(BORDER,480+BORDER*3.5))

        waypoint_text = "[WAYPOINTS] "
        if(WAYPOINTS.shape[0]==1):
            waypoint_text+="click map to add waypoint, BACKSPACE to discard last, DELETE to clear all"
            waypoint_text = font.render(waypoint_text,True,(255,255,255))

        else:
            for coords in WAYPOINTS:
                waypoint_text+=f"{coords[0]:.1f} {coords[1]:.1f} | "
            waypoint_text = font.render(waypoint_text,True,green)
        screen.blit(waypoint_text,(BORDER,480+BORDER*5))

        instructions = font.render("[CONTROLS] move: wasd | sprint: space | run_waypoint: ENTER | zoom in: m | zoom out: n",True,(255,255,255),blue)
        screen.blit(instructions,(BORDER,480+BORDER*6.5))

        screen = pygame.transform.scale(screen, (window.get_width() , window.get_width()*screen_height/screen_width ))
        window.blit(screen, (0, 0))
        pygame.display.flip()