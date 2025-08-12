import numpy as np
import cv2
import matplotlib.pyplot as mpimg

"""
# Import pandas and read in csv file as a dataframe
import pandas as pd
# Change the path below to your data directory
# If you are in a locale (e.g., Europe) that uses ',' as the decimal separator
# change the '.' to ','
df = pd.read_csv('../test_dataset/robot_log.csv', delimiter=';', decimal='.')
csv_img_list = df["Path"].tolist() # Create list of image pathnames
# Read in ground truth map and create a 3-channel image with it
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(float)


# Creating a class to be the data container
# Will read in saved data from csv file and populate this object
# Worldmap is instantiated as 200 x 200 grids corresponding 
# to a 200m x 200m space (same size as the ground truth map: 200 x 200 pixels)
# This encompasses the full range of output position values in x and y from the sim
class Databucket():
    def __init__(self):
        self.images = csv_img_list  
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        self.count = 0 # This will be a running index
        self.worldmap = np.zeros((200, 200, 3)).astype(float)
        self.ground_truth = ground_truth_3d # Ground truth worldmap

# Instantiate a Databucket().. this will be a global variable/object
# that you can refer to in the process_image() function below
data = Databucket()
"""


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float32)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float32)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask


def find_rocks(img, thresh1 = (120, 110, 40)):
    color_thresh = np.zeros_like(img[:, :, 2])
    #thresh1 = (10, 50, 70)
    #120)
    above_thresh = (img[:, :, 0] > thresh1[0]) \
    & (img[:, :, 1] > thresh1[1]) \
    & (img[:, :, 2] < thresh1[2])
    #below_thresh = (thresh2[0] > img[:, :, 0]) \
    ##& (thresh2[1] > img[:, :, 1]) \
    #& (thresh2[2] > img[:, :,2])
    color_thresh[above_thresh] = 1
    #color_thresh[below_thresh] = 0
    return color_thresh


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles


    img = Rover.img
    xpos, ypos = Rover.pos[0], Rover.pos[1]
    yaw = Rover.yaw
    shape = img.shape
    bottom_offset = 6
    dst_size = 5
    scale = 10
    world_size = Rover.worldmap.shape[0]

    # Solution 01
    # Source Points
    source = np.float32(
        [
            [13.7, 140.0],
            [301.1, 139.3],
            [198.9, 95.9],
            [118.0, 95.7]
        ]
    )
    #Destination points
    destination = np.float32(
        [
            [shape[1]/2 - dst_size, shape[0] - bottom_offset],
            [shape[1]/2 + dst_size, shape[0] - bottom_offset],
            [shape[1]/2 + dst_size, shape[0] - bottom_offset - dst_size * 2],
            [shape[1]/2 - dst_size, shape[0] - bottom_offset - dst_size * 2]
        ]
    )
    
    #Solution 02
    # Perspective transform
    warped, masked = perspect_transform(Rover.img, source, destination)

    # Identifying obstacle map
    

    # Solution 03
    # Navigable terrain
    threshed = color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed)-1 * masked)
    # Rock Samples
    #rock_samples = find_rocks(img)
    Rover.vision_image[:, :, 2] = threshed * 255
    Rover.vision_image[:, :, 0] = obs_map * 255

    
    xpix, ypix = rover_coords(threshed)

    x_world, y_world = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)

    obsxpix, obsypix = rover_coords(obs_map)

    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, xpos, ypos, yaw, world_size, scale)


    # Solution 07:
    Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
    Rover.worldmap[y_world, x_world, 2] += 10

    #nav_pix = Rover.worldmap[:, :, 2] > 0
    #Rover.worldmap[nav_pix, 0] = 0
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_angles = angles

    rock_map = find_rocks(warped, (110, 110, 50))

    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, world_size, scale)
        
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]
        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        Rover.vision_image[:, :, 1] = rock_map * 255
    else:
        Rover.vision_image[:, :, 1] = 0
    

        #data.worldmap[rock_y_world, rock_x_world, 1] = 255


        """
    
    output_image = np.zeros((img.shape[0] + Rover.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

        # Let's create more images to add to the mosaic, first a warped image
    warped, masked = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(Rover.worldmap, 1, Rover.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:Rover.worldmap.shape[1]] = np.flipud(map_add)
    
    """

    return Rover