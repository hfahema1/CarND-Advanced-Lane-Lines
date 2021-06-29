import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# --- CAMERA CALIBRATION ---
def camera_calibration(calibrationDirectory):
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

    # # run an adaptive thresholding on S channel
    adaptive_blur = cv2.GaussianBlur(S,(5,5),0)
    adaptive_gaus = cv2.adaptiveThreshold(adaptive_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,5)
    adaptive_sobelx = cv2.Sobel(adaptive_gaus, cv2.CV_64F,dx = 1, dy = 0) 
    adaptiveabs_sobelx = np.absolute(adaptive_sobelx)
    adaptive_scaled_sobel = np.uint8(255*adaptiveabs_sobelx/np.max(adaptiveabs_sobelx))
    gausbinary = np.zeros_like(adaptive_scaled_sobel)
    gausbinary[(adaptive_scaled_sobel >= x_thresh_min) & (adaptive_scaled_sobel <= x_thresh_max)] = 1

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
    
    # combine thresholded and masked images
    combined = np.zeros_like(binary_dir)
    combined[ (binary_sat == 1) | ((binary_mag == 1) & (binary_dir == 1)) ]= 1

    return combined

def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    imgShape = img.shape
    
    vertices = np.array([[((.40*imgShape[1]), .68*imgShape[0]),
                          ((.60*imgShape[1]), .68*imgShape[0]),
                          ((.95*imgShape[1]), imgShape[0]),
                          ((.05*imgShape[1]), imgShape[0])]],
                          dtype=np.int32)
    #defining a 3 channel to fill the mask
    ignore_mask_color = (255,)
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    ax5.set_title("region of interest")
    ax5.imshow(cv2.bitwise_and(img, mask))

    #returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)

# --- PERSPECTIVE TRANSFORM TO "BIRDS-EYE VIEW" ---
def warp(image):
    shape_y = image.shape[0]
    shape_x = image.shape[1]
    img = region_of_interest(image)
    # define 4 source points 
    src = np.float32(
        [[.4*shape_x,.68*shape_y],
        [.6*shape_x,.68*shape_y],
        [.95*shape_x, shape_y],
        [.05*shape_x, shape_y]])

    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32(
        [[.11*shape_x,0],
        [.89*shape_x,0],
        [.90*shape_x,shape_y],
        [.10*shape_x,shape_y]])

    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src,dst)
    # compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)
    
    return warped, Minv

# --- DETECT LANE PIXELS AND FIT TO FIND THE LANE BOUNDARY ---
def search_around_poly(binary_warped):
    print('inside poly search')
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
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def find_lane_pixels(binary_warped):
    print('inside window search')
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
    margin = 50
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

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # --- DEBUG PLOTS ---
    # Draw the windows on the visualization image
    ax6.set_title("Windows")
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
                    /(2.0*np.absolute(left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1+(2.0*right_fit_cr[0]*y_eval+right_fit_cr[1])**2)**(1.5)) \
                    /(2.0*np.absolute(right_fit_cr[0]))  ## Implement the calculation of the right line here
    
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
    cv2.putText(img2write,
        'L Curve Rad = {curve:.2f}m'.format(curve = left_curverad), 
        (600,30), 
        font, 
        fontScale,
        fontColor,
        lineType)
    cv2.putText(img2write,
        'R Curve Rad = {curve:.2f}m'.format(curve = right_curverad), 
        (600,60), 
        font, 
        fontScale,
        fontColor,
        lineType)
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
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = None
        # y values of the last n fits of the line
        self.recent_yfitted = None
        #x values of average polynomial coeffs fitted line over the last n iterations
        self.bestx = None     
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
        print('self.allx = ', len(self.allx))
        self.detected = len(self.allx)>10000
        print('len(self.allx)',len(self.allx))
        print(self.detected) 

    # get poly coeffs from x and y pixels. 
    # add coeffs to list
    def set_current_fit(self):
        self.current_fit = np.polyfit(self.ally,self.allx,2)
        self.add_best_fit()

    # add poly coeffs to list of coeffs and delete oldest set if list > 5.
    # run the averager
    def add_best_fit(self):
        print('1.', self.best_fit)
        if (self.best_fit[0].any() == False):
            self.best_fit = [self.current_fit]
        else:
            print('2', self.current_fit)
            self.best_fit = np.append(self.best_fit,[self.current_fit],axis=0)
            print('3.',self.best_fit)
            if (self.best_fit.shape[0] > 10):
                self.best_fit = np.delete(self.best_fit,0,0)
                print('4.',self.best_fit)
        self.set_bestx()

    # take average of last 5 coeffs
    # run a fitted line averaged coeffs
    def set_bestx(self):
        self.bestx = np.mean(self.best_fit,axis=0)
        print('5.',self.bestx)
        self.set_recent_xfitted()

    # use lin space from img size and average coeffs to product plot lines 
    def set_recent_xfitted(self):
        self.recent_xfitted = self.bestx[0]*self.recent_yfitted**2 \
                            + self.bestx[1]*self.recent_yfitted \
                            + self.bestx[2]
        self.recent_xfitted = np.clip(self.recent_xfitted, a_min = 0, a_max = 1279)        

# --- PIPELINE ---
# compute the camera calibration matrix and distortion 
#           coefficients given a set of chessboard images.
camera_calibration('camera_cal/')

# instantiate line classes for left and right lines
rLine = Line()
lLine = Line()

f, ((ax1, ax2, ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, figsize=(20,10))
def process_image(image):
     # distortion correction to raw images.
    undist = distortion_correction(image,'camera_cal/')
    # color transforms, gradients, etc., to create a thresholded binary image.
    thresholded = thresholds(undist)
    # perspective transform to rectify binary image
    top_down, Minv = warp(thresholded)
    # detect lane pixels. Find pixels if last detection failed
    print("is either line detection false?")
    print((rLine.detected == False) or (rLine.detected == False))
    if ((rLine.detected == False) or (rLine.detected == False)):
        print("running window search")
        left_x, left_y, right_x, right_y = find_lane_pixels(top_down)
    else:
        print("running poly search")
        left_x, left_y, right_x, right_y = search_around_poly(top_down)

    # assign lane pixels to lane objects and find lane boundaryies
    lLine.set_pixels(left_x, left_y, top_down.shape)
    rLine.set_pixels(right_x, right_y, top_down.shape)

    # Determine the curvature of the lane and vehicle position with respect to center.
    lLine.radius_of_curvature, rLine.radius_of_curvature, delta_center = measure_curvature_real(lLine.allx, lLine.ally, rLine.allx, rLine.ally)
    # Warp the detected lane boundaries back onto the original image.
    img_with_lanes = unwarp(undist,top_down.shape, lLine.recent_xfitted, rLine.recent_xfitted,Minv)
    # write curvature data onto image
    img_with_data = write_img_data(img_with_lanes, lLine.radius_of_curvature, rLine.radius_of_curvature, delta_center)
    
    # --- DEBUG SUBPLOTTING ---
    ax1.set_title('thresholded')
    ax1.imshow(thresholded)
    ax2.set_title('top_down')
    ax2.imshow(top_down)
    ax3.set_title('img_with_lanes')
    ax3.imshow(img_with_lanes)
    ax4.set_title('top_down w/ poly')
    ax4.imshow(top_down)
    lslope_str = 'L Slope = {slope:.2f}'.format(slope = (lLine.recent_xfitted[0]-lLine.recent_xfitted[-1]) / 
                                (lLine.recent_yfitted[0]-lLine.recent_yfitted[-1]))
    rslope_str = 'R Slope = {slope:.2f}'.format(slope = (rLine.recent_xfitted[0]-rLine.recent_xfitted[-1]) / 
                                (rLine.recent_yfitted[0]-rLine.recent_yfitted[-1]))
    ax4.text(0,40,lslope_str,color='white')
    ax4.text(0,80,rslope_str,color='white')
    ax4.plot(lLine.recent_xfitted,lLine.recent_yfitted, color = 'red')
    ax4.plot(rLine.recent_xfitted,rLine.recent_yfitted, color = 'red')
    plt.show()
    

    
    return img_with_data

#--- LET'S MAKE A VIDEO ---
project_output = 'project_video_OutputSubclip.mp4'
clip1 = VideoFileClip("project_video.mp4")#.subclip(30.00,33.00)
project_clip = clip1.fl_image(process_image)
#clip1.write_images_sequence('image_sequence/frame%04d.jpeg',verbose = False)
project_clip.write_videofile(project_output, audio=False)


# # # -- TEST WITH IMAGE --
test_pic = mpimage.imread('image_sequence/frame0074.jpeg')
processed_image = process_image(test_pic)