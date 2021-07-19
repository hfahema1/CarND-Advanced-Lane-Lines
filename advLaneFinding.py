import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pickle
from moviepy.editor import VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage

# --- CAMERA CALIBRATION ---
def camera_calibration(calibrationDirectory):
    # prepare subplot for camera calibration
    global ax9
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    # arrays to store object points and image points from all the images.
    objPoints = [] # 3d points in real world space
    imgPoints = [] # 2d points in image plane.
    # make list of calibration images and get image size
    images = glob.glob(calibrationDirectory + "cal*.jpg")
    # step through list and search for chessboard corners
    for idx, fileName in enumerate(images):
        img = mpimage.imread(fileName)
        imgSize = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points
        if ret == True:
            objPoints.append(objp)
            imgPoints.append(corners)
            if str(fileName) == 'camera_cal\calibration3.jpg':
                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                ax9.clear()
                ax9.imshow(img)
    # camera calibration w/ completed object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, imgSize,None,None)
    # save the camera calibration result for later use
    distPickle = {}
    distPickle["mtx"] = mtx
    distPickle["dist"] = dist
    distPickle["rvecs"] = rvecs
    distPickle["tvecs"] = tvecs
    pickle.dump( distPickle, open( calibrationDirectory + "distPickle.p", "wb" ) )

# --- APPLY DISTORTION CORRECTION ---
def distortion_correction(img,calibrationDirectory):
    # read in the saved camera matrix and distortion coefficients from pickle file
    distPickle = pickle.load( open( calibrationDirectory + "distPickle.p", "rb" ) )
    mtx = distPickle["mtx"]
    dist = distPickle["dist"]
    # undistort using mtx and dist
    unDist = cv2.undistort(img, mtx, dist, None, mtx)
    return unDist

# --- CREATE THRESHOLDED BINARY IMAGE W/ GRADIENTS, COLOR TRANSFORMS, ETC. ---
def thresholds(image):
    # Convert image to color spaces that are useful
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    S = hls[:,:,2]
    # threshold mins and maxs
    x_thresh_min=20
    x_thresh_max=100
    s_thresh_min=150
    s_thresh_max=255
    mag_thresh_min=30
    mag_thresh_max=100
    dir_thresh_min = .7
    dir_thresh_max = 1.3
    # run a Sobel x on the S value. take abs value and scale
    satSobelX = cv2.Sobel(S, cv2.CV_64F, 1, 0)
    absSatSobelX = np.absolute(satSobelX)
    scaledSatSobelX = np.uint8(255*absSatSobelX/np.max(absSatSobelX))
    # threshold S value
    binary_sat = np.zeros_like(scaledSatSobelX)
    binary_sat[(S >= s_thresh_min) & (S <= s_thresh_max)] = 1
    # run Sobel x and y on the GRAY value. take abs value
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    # calc magnitude and scale
    mag_sobel = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scaled_sobel_mag = np.uint8(255*mag_sobel/np.max(mag_sobel))
    # threshold magnitude
    binary_mag = np.zeros_like(scaled_sobel_mag)
    binary_mag[(scaled_sobel_mag >= mag_thresh_min) & (scaled_sobel_mag <= mag_thresh_max)] = 1
    # calc direction of gradient
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # threshold direction
    binary_dir = np.zeros_like(dir_sobel)
    binary_dir[(dir_sobel >= dir_thresh_min) & (dir_sobel <= dir_thresh_max)] = 1
    # return combined thresholded images
    combined = np.zeros_like(binary_dir)
    combined[ (binary_sat == 1) | ((binary_mag == 1) & (binary_dir == 1)) ]= 1
    return combined

def region_of_interest(img):
    # prepare subplot for image masking
    global ax3
    # defining a blank mask to start with
    mask = np.zeros_like(img)   
    imgShape = img.shape
    vertices = np.array([[((.40*imgShape[1]), .63*imgShape[0]),
                          ((.60*imgShape[1]), .63*imgShape[0]),
                          ((1.0*imgShape[1]), imgShape[0]),
                          ((.00*imgShape[1]), imgShape[0])]],
                          dtype=np.int32)
    # defining single channel to fill the mask
    ignore_mask_color = (255,)
    # filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # plot mask image
    ax3.clear()
    ax3.imshow(cv2.bitwise_and(img, mask))
    #returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)

# --- PERSPECTIVE TRANSFORM TO "BIRDS-EYE VIEW" ---
def warp(img):
    shape_y = img.shape[0]
    shape_x = img.shape[1]
    # define 4 source points 
    src = np.float32(
        [[.44*shape_x,.63*shape_y],
        [.56*shape_x,.63*shape_y],
        [.95*shape_x, shape_y],
        [.05*shape_x, shape_y]])
    # define 4 destination points
    dst = np.float32(
        [[.035*shape_x,0],
        [.965*shape_x,0],
        [.90*shape_x,shape_y],
        [.10*shape_x,shape_y]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src,dst)
    # compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (shape_x,shape_y), flags=cv2.INTER_LINEAR)
    # return warped image and inverse matrix
    return warped, Minv

# --- DETECT LANE PIXELS AND FIT TO FIND THE LANE BOUNDARY ---
def search_poly(binary_warped):
    # prepare subplot for polynomial search
    global ax7
    # create image to visualize result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Set margin and grab activated pixels
    margin = 50
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the area of search based on activated x-values ###
    left_lane_inds = ( ( nonzerox > (lLine.bestx[0] *(nonzeroy**2) + 
                                     lLine.bestx[1] * nonzeroy + 
                                     lLine.bestx[2] - margin) ) & 
                       ( nonzerox < (lLine.bestx[0] *(nonzeroy**2) + 
                                     lLine.bestx[1] * nonzeroy +
                                     lLine.bestx[2] + margin) )    )
    right_lane_inds = ( (nonzerox > (rLine.bestx[0] *(nonzeroy**2) + 
                                     rLine.bestx[1] * nonzeroy +
                                     rLine.bestx[2] - margin) ) &
                       ( nonzerox < (rLine.bestx[0]*(nonzeroy**2) +
                                     rLine.bestx[1]*nonzeroy +
                                     rLine.bestx[2] + margin)))
    # extract left and right line pixel positions and return them
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([lLine.recent_xfitted-margin, lLine.recent_yfitted]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([lLine.recent_xfitted+margin, 
                              lLine.recent_yfitted])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([rLine.recent_xfitted-margin, rLine.recent_yfitted]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rLine.recent_xfitted+margin, 
                              rLine.recent_yfitted])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plot image onto subplot
    ax7.clear()
    ax7.imshow(out_img)
    return leftx, lefty, rightx, righty

def search_windows(binary_warped):
    # prepare subplot to show image
    global ax6
    # histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # create image to visualize result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    #  peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # number of sliding windows, width of the windows +/- margin, minimum number of pixels found to recenter window
    nwindows = 9
    margin = 75
    minpix = 50
    # height of windows based on nwindows and image shape
    window_height = int(binary_warped.shape[0]//nwindows)
    # x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # empty lists to receive left and right lane pixel indices
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
        # draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)
        # identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # add these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # found > minpix pixels, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    # concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error
        pass
    # extract left and right line pixel positions and return them
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # plot image on subplot
    ax6.clear()
    ax6.imshow(out_img)

    return leftx, lefty, rightx, righty

# --- DETERMINE CURVATURE OF THE LANE AND VEHICLE POSITION WRT CENTER ---
def measure_curvature_real(leftx, lefty, rightx, righty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/860 # meters per pixel in x dimension
    # reverse to match top-to-bottom in y
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # measure curavture from image bottom
    y_eval = np.max(righty)*ym_per_pix
    # subtract image midpoint from polynomials' midpoint for distance from center
    delta_center = ((rightx[-1]+leftx[-1])/2) - (1280/2)
    delta_center = delta_center*xm_per_pix
    # implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval+left_fit_cr[1])**2)**(1.5)) \
                    /(2.0*np.absolute(left_fit_cr[0]))
    right_curverad = ((1+(2.0*right_fit_cr[0]*y_eval+right_fit_cr[1])**2)**(1.5)) \
                    /(2.0*np.absolute(right_fit_cr[0]))
    # return measurements
    return left_curverad, right_curverad, delta_center

# --- WARP THE DETECED LANE BOUNDARIES BACK ONTO THE ORIGINAL IMAGE ---
def unwarp(original,img_shape, left_fitx, right_fitx, Minv,α=1., β=.3, γ=0.):
    # create an image to draw the lines on
    warp_zero = np.zeros_like(original).astype(np.uint8)
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
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
    # define text style parameters
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    # write left line data onto image
    cv2.putText(img2write,
        'L Curve Rad = {curve:.2f}m'.format(curve = left_curverad), 
        (600,30), 
        font, 
        fontScale,
        fontColor,
        lineType)
    # write right line data onto image
    cv2.putText(img2write,
        'R Curve Rad = {curve:.2f}m'.format(curve = right_curverad), 
        (600,60), 
        font, 
        fontScale,
        fontColor,
        lineType)
    # write center position data onto image
    cv2.putText(img2write,
        'Off Center = {center:.2f}m'.format(center = delta_center), 
        (600,90), 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img2write

# --- LINE CLASS ---
class Line():
    def __init__(self, name):
        # name of line used for debugging
        self.name = name
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # y values of the last n fits of the line
        self.recent_yfitted = None
        #x values of average polynomial coeffs fitted line over the last n iterations
        self.bestx = np.array([0,0,0], dtype='float')     
        #polynomial coefficients of the last n iterations
        self.best_fit = [np.array([False])]
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # position at base of image
        self.base = None
    # assign detected pixels to lane object and assign y values for fit line
    # run a fit and check for line detection
    def set_pixels(self,x,y,img_shape):
        self.allx = x
        self.ally = y
        self.recent_yfitted = np.linspace(0, img_shape[0]-1, img_shape[0])
        self.set_detected()
        self.set_current_fit()
    # set if number of pixels detected is greater than 50
    def set_detected(self):
        margin = 5000
        self.detected = len(self.allx) > margin
    # get poly coeffs from x and y pixels. 
    # add coeffs to list
    def set_current_fit(self):
        self.current_fit = np.polyfit(self.ally,self.allx,2)
        #self.check_diffs()
        self.add_best_fit()
    # set difference from current coeffs to past coeffs (NOT USED)
    def check_diffs(self):
        self.diffs = self.current_fit - self.bestx
        if (self.bestx[0] != 0):
            coef0_delta = self.current_fit[0]/self.bestx[0]
            coef1_delta = self.current_fit[1]/self.bestx[1]
            coef2_delta = self.current_fit[2]/self.bestx[2]
            if coef0_delta > 5:
                self.current_fit[0] = 5.*self.bestx[0]
            elif coef0_delta < -5:
                self.current_fit[0] = -5.*self.bestx[0]
            if coef1_delta > 2:
                self.current_fit[1] = 2.*self.bestx[1]
            elif coef1_delta < -5:
                self.current_fit[1] = -2.*self.bestx[1]
            if coef2_delta > 1.001:
                self.current_fit[2] = 1.001*self.bestx[2]
            elif coef2_delta < -1.001:
                self.current_fit[2] = -1.001*self.bestx[2]
    # append to size limited best_fit list of coeffs
    # and run averager
    def add_best_fit(self):
        set_length = 10
        if (self.best_fit[0].any() == False):
            self.best_fit = [self.current_fit]
        else:
            self.best_fit = np.append(self.best_fit,[self.current_fit],axis=0)
            if (self.best_fit.shape[0] > set_length):
                self.best_fit = np.delete(self.best_fit,0,0)
        self.set_bestx()
    # take average best_fit coeffs
    # and run a fitted line
    def set_bestx(self):
        self.bestx = np.mean(self.best_fit,axis=0)
        self.set_recent_xfitted()
    # use stored lin space (from img size) and average coeffs to product lines 
    # clip to image boundary
    def set_recent_xfitted(self):
        self.recent_xfitted = self.bestx[0]*self.recent_yfitted**2 \
                            + self.bestx[1]*self.recent_yfitted \
                            + self.bestx[2]
        self.recent_xfitted = np.clip(self.recent_xfitted, a_min = 0, a_max = 1279)        

# --- PROCESS PLOTS ---
def process_plotting(thresholded, top_down, img_with_data, undist):
    global f, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9
    ax1.clear()
    ax1.set_title('undist')
    ax1.imshow(undist)
    ax2.clear()
    ax2.set_title('thresholded')
    ax2.imshow(thresholded)
    ax3.set_title("region of interest")
    ax4.clear()
    ax4.set_title('top_down')
    ax4.imshow(top_down)
    ax5.clear()
    ax5.set_title('top_down w/ poly')
    ax5.imshow(top_down)
    ax5.plot(lLine.recent_xfitted,lLine.recent_yfitted, color = 'red')
    ax5.plot(rLine.recent_xfitted,rLine.recent_yfitted, color = 'red')
    ax6.set_title("Search Windows")
    ax7.set_title("Search Poly")
    ax8.clear()
    ax8.set_title('img_with_data')
    ax8.imshow(img_with_data)
    ax9.set_title('camera calibration')
    plt.show()
    # return figure with all subplots
    return f

# --- PIPELINE ---
# prepare plots to display image process
f, ((ax1, ax2, ax3),(ax4, ax5, ax6),(ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24, 9))
# compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
camera_calibration('camera_cal/')
# instantiate line classes for left and right lines
rLine = Line('right')
lLine = Line('left')
# run pipeline
def process_image(image):
    # figure for plotting process
    global f
    # distortion correction to raw images.
    undist = distortion_correction(image,'camera_cal/')
    # color transforms, gradients, etc., to create a thresholded binary image.
    thresholded = thresholds(undist)
    # mask the image and retain only lane area
    masked = region_of_interest(thresholded)
    # perspective transform to rectify binary image
    top_down, Minv = warp(masked)
    # Detect lane pixels. Find pixels with polynomial if last attempt succeeded
    if (rLine.detected and lLine.detected) :
        left_x, left_y, right_x, right_y = search_poly(top_down)
    else :
        left_x, left_y, right_x, right_y = search_windows(top_down)
    # assign lane pixels to lane objects and find lane boundaryies
    lLine.set_pixels(left_x, left_y, top_down.shape)
    rLine.set_pixels(right_x, right_y, top_down.shape)
    # Determine the curvature of the lane and vehicle position with respect to center.
    lLine.radius_of_curvature, rLine.radius_of_curvature, delta_center = measure_curvature_real(lLine.allx, lLine.ally, rLine.allx, rLine.ally)
    # Warp the detected lane boundaries back onto the original image.
    img_with_lanes = unwarp(undist,top_down.shape, lLine.recent_xfitted, rLine.recent_xfitted,Minv)
    # write curvature data onto image
    img_with_data = write_img_data(img_with_lanes, lLine.radius_of_curvature, rLine.radius_of_curvature, delta_center)
    
    # --- PROCESS PLOTS (UNCOMMENT TO USE)---
    process_plotting(thresholded, top_down, img_with_data, undist)
    # return the processed image
    return img_with_data
    # return subplots as image. comment out the above return to use
    # takes a VERY long time
    return mplfig_to_npimage(f)

#--- LET'S MAKE A VIDEO ---
# project_output = 'project_video_Output.mp4'
# clip1 = VideoFileClip("project_video.mp4")#.subclip(38.00,42.00)
# project_clip = clip1.fl_image(process_image)
# #clip1.write_images_sequence('image_sequence/frame%04d.jpeg',verbose = False)
# project_clip.write_videofile(project_output, audio=False)

# -- TEST WITH IMAGE --
test_pic = mpimage.imread('test_images/straight_lines1.jpg')
processed_image = process_image(test_pic)