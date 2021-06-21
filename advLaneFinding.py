# Compute the camera calibration matrix and distortion 
#           coefficients given a set of chessboard images.

# Apply a distortion correction to raw images.

# Use color transforms, gradients, etc., to create a thresholded 
#           binary image.

# Apply a perspective transform to rectify binary image 
#           ("birds-eye view").

# Detect lane pixels and fit to find the lane boundary.

# Determine the curvature of the lane and vehicle position 
#           with respect to center.

# Warp the detected lane boundaries back onto the original image.

# Output visual display of the lane boundaries and numerical 
#           estimation of lane curvature and vehicle position.


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# --- CAMERA CALIBRATION ---
def camera_calibration(calibration_directory):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images and get image size
    images = glob.glob(calibration_directory + "cal*.jpg")
    img_size = ()

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = mpimage.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            write_name = 'corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # plt.imshow(img)
            # plt.show()
            # cv2.waitKey(500)

    cv2.destroyAllWindows()
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                    img_size,None,None)

    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    pickle.dump( dist_pickle, open( calibration_directory+"dist_pickle.p", "wb" ) )

# --- APPLY DISTORTION CORRECTION ---
def distortion_correction(img,calibration_directory):
    # read in the saved camera matrix and distortion coefficients
    dist_pickle = pickle.load( open( calibration_directory + "dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

# --- CREATE THRESHOLDED BINARY IMAGE W/ GRADIENTS, COLOR TRANSFORMS, ETC. ---
def thresholds(img,s_thresh_min=150,s_thresh_max=255,x_thresh_min=20,x_thresh_max=100):
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    imshape = img.shape
    # run a Sobel x on the saturation value of the HLS colorspace
    # take the derivative in x
    satsobelx = cv2.Sobel(S, cv2.CV_64F, 1, 0)
    # absolute x derivative to accentuate lines away from horizontal
    satabs_sobelx = np.absolute(satsobelx)
    satscaled_sobel = np.uint8(255*satabs_sobelx/np.max(satabs_sobelx))

    # Threshold x gradient w/ other color space
    satsxbinary = np.zeros_like(satscaled_sobel)
    satsxbinary[(satscaled_sobel >= x_thresh_min) & (satscaled_sobel <= x_thresh_max)] = 1

    # apply a threshold to the S channel
    sat_binary = np.zeros_like(satscaled_sobel)
    sat_binary[(S >= s_thresh_min) & (S <= s_thresh_max)] = 1

    # apply a mask to lane area
    vertices = np.array([[(.41*imshape[1], .63*imshape[0]),   # bottom left
                          (.59*imshape[1], .63*imshape[0]),   # top left
                          (.89*imshape[1], imshape[0]),   # top right
                          (.11*imshape[1], imshape[0])]], # bottom right
                          dtype=np.int32)

    masked_binary = region_of_interest(sat_binary, vertices)

    return masked_binary

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    # 3) Calculate the magnitude 
    abs_sobelxy = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


# --- PERSPECTIVE TRANSFORM TO "BIRDS-EYE VIEW" ---
def warp(img):
    shape_y = img.shape[0]
    shape_x = img.shape[1]

    # define 4 source points 
    src = np.float32(
        [[.465*shape_x,.64*shape_y],
        [.535*shape_x,.64*shape_y],
        [.74*shape_x, .97*shape_y],
        [.26*shape_x, .97*shape_y]])

    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32(
        [[.25*shape_x,0],
        [.75*shape_x,0],
        [.75*shape_x,shape_y],
        [.25*shape_x,shape_y]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src,dst)

    # compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped, Minv

# --- DETECT LANE PIXELS AND FIT TO FIND THE LANE BOUNDARY ---
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 40
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def left_running_avg(running_list, new_item = None):
    left_a.append(running_list[2])
    left_b.append(running_list[1])
    left_c.append(running_list[0])
    if (len(left_a) > 10):
        left_a.pop(0)
    if (len(left_b) > 10):
        left_b.pop(0)
    if (len(left_c) > 10):
        left_c.pop(0)
    a_avg = sum(left_a)/len(left_a)
    b_avg = sum(left_b)/len(left_b)
    c_avg = sum(left_c)/len(left_c)

    return c_avg, b_avg, a_avg

def right_running_avg(running_list, new_item = None):
    right_a.append(running_list[2])
    right_b.append(running_list[1])
    right_c.append(running_list[0])
    if (len(right_a) > 10):
        right_a.pop(0)
    if (len(right_b) > 10):
        right_b.pop(0)
    if (len(right_c) > 10):
        right_c.pop(0)
    a_avg = sum(right_a)/len(right_a)
    b_avg = sum(right_b)/len(right_b)
    c_avg = sum(right_c)/len(right_c)
    print('right a = ')
    print(right_a)
    return c_avg, b_avg, a_avg, 

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    # left_fit = np.polyfit(lefty,leftx,deg=2)
    # right_fit = np.polyfit(righty,rightx,deg=2)
    left_fit = np.polynomial.polynomial.polyfit(lefty,leftx,2)
    right_fit= np.polynomial.polynomial.polyfit(righty,rightx,2)
    left_fit_avg = left_running_avg(left_fit)
    right_fit_avg = right_running_avg(right_fit)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit_avg[2]*ploty**2 + left_fit_avg[1]*ploty + left_fit_avg[0]
        right_fitx = right_fit_avg[2]*ploty**2 + right_fit_avg[1]*ploty + right_fit_avg[0]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty


# --- DETERMINE CURVATURE OF THE LANE AND VEHICLE POSITION WRT CENTER ---
def measure_curvature_real(img_size,leftx,rightx,ploty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)*ym_per_pix

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # subtract image midpoint from polynomials' midpoint for distance from center
    delta_center = (rightx[-1]-leftx[-1])-img_size[0]/2
    delta_center = delta_center*xm_per_pix
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval+left_fit_cr[1])**2)**(1.5)) \
                    /(2.0*np.absolute(left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1+(2.0*right_fit_cr[0]*y_eval+right_fit_cr[1])**2)**(1.5)) \
                    /(2.0*np.absolute(right_fit_cr[0]))  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad, delta_center

# --- WARP THE DETECED LANE BOUNDARIES BACK ONTO THE ORIGINAL IMAGE ---
def unwarp(original ,left_fitx, right_fitx, ploty,Minv,α=1., β=.3, γ=0.):
    # create an image to draw the lines on
    warp_zero = np.zeros_like(original).astype(np.uint8)
    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, Minv, (original.shape[1], original.shape[0]))

    return cv2.addWeighted(original, α, newwarp, β, γ)

# --- OUTPUT VISUAL DISPLAY OF THE LANE BOUNDARIES, LANE CURVATURE AND ---
# --- VEHICLE POSITION WITH RESPECT TO CENTER --- 
def write_img_data(img2write,left_curverad, right_curverad, delta_center):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(img2write,
        'L Curve Rad = ' + str(left_curverad), 
        (600,30), 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.putText(img2write,
        'R Curve Rad = ' + str(right_curverad), 
        (600,60), 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.putText(img2write,
        'Off Center = ' + str(delta_center), 
        (600,90), 
        font, 
        fontScale,
        fontColor,
        lineType)

    return img2write

# --- LINE CLASS ---
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

# --- PIPELINE ---
# compute the camera calibration matrix and distortion 
#           coefficients given a set of chessboard images.
camera_calibration('camera_cal/')

left_a = []
left_b = []
left_c = []

right_a = []
right_b = []
right_c = []

def process_image(image):
    # instantiate line classes for left and right lines
    right_line = Line
    left_line = Line
    # apply a distortion correction to raw images.
    undist = distortion_correction(image,'camera_cal/')
    # ase color transforms, gradients, etc., to create a thresholded 
    #           binary image.
    sobel_x_sat = thresholds(undist)
    mag_binary = mag_thresh(undist, mag_thresh=(30, 100))
    dir_binary = dir_threshold(undist, thresh=(.7, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[(sobel_x_sat == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # Apply a perspective transform to rectify binary image 
    #           ("birds-eye view").
    top_down, Minv = warp(combined)
    # Detect lane pixels and fit to find the lane boundary.
    top_down_poly, left_fitx, right_fitx, ploty = fit_polynomial(top_down)
    # Determine the curvature of the lane and vehicle position 
    #           with respect to center.
    left_line.radius_of_curvature, right_line.radius_of_curvature, delta_center = measure_curvature_real(top_down_poly.shape,
                                                                        left_fitx, 
                                                                        right_fitx, 
                                                                        ploty)
    # Warp the detected lane boundaries back onto the original image.
    img_with_lanes = unwarp(undist,left_fitx, right_fitx, ploty,Minv)
    # Output visual display of the lane boundaries and numerical 
    #           estimation of lane curvature and vehicle position.
    img_with_data = write_img_data(img_with_lanes,left_line.radius_of_curvature, right_line.radius_of_curvature, delta_center)
    
    f, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24, 9))
    ax1.imshow(undist)
    ax1.set_title('undist', fontsize=20)
    ax2.imshow(sobel_x_sat)
    ax2.set_title('sobel_x_sat', fontsize=20)
    ax3.imshow(mag_binary)
    ax3.set_title('mag_binary', fontsize=20)
    ax4.imshow(dir_binary)
    ax4.set_title('dir_binary', fontsize=20)
    ax5.imshow(combined)
    ax5.set_title('combined', fontsize=20)
    ax6.imshow(top_down)
    ax6.set_title('top_down', fontsize=20)
    ax7.imshow(top_down_poly)
    ax7.set_title('top_down_poly', fontsize=20)
    ax8.imshow(top_down_poly)
    ax8.plot(left_fitx, ploty, color = 'yellow')
    ax8.plot(right_fitx, ploty, color = 'yellow')
    ax8.set_title('top_down_poly', fontsize=20)
    ax9.imshow(img_with_data)
    ax9.set_title('img_with_data', fontsize=20)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    return img_with_data

# project_output = 'harderChallengeOutput.mp4'
# clip1 = VideoFileClip("harder_challenge_video.mp4")#.subclip(23.00,25.00)
# #clip1.write_images_sequence('image_sequence/frame%04d.jpeg',verbose = False)
# project_clip = clip1.fl_image(process_image)
# project_clip.write_videofile(project_output, audio=False)

test_pic = mpimage.imread('test_images\straight_lines1.jpg')
processed_image = process_image(test_pic)

