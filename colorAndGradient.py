import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# Plotting thresholded images
f, ((ax1, ax2, ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, figsize=(20,10))

# Read in an image, you can also try test1.jpg or test4.jpg
img = mpimg.imread('test_images/test1.jpg') 

# Separate color space channels
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
s_channel = hls[:,:,2]
lower_yellow = np.array([20,0,0])
upper_yellow = np.array([40,255,255])
lower_white = np.array([0,0,0])
upper_white = np.array([255,30,255])
lower_white2 = np.array([0,235,0])
upper_white2 = np.array([255,255,255])

r_channel = img[:,:,0]
# Threshold x gradient
x_thresh_min = 20
x_thresh_max = 100
# Threshold S Channel
s_thresh_min = 150
s_thresh_max = 255
# Threshold R channel
r_thresh_min = 200
r_thresh_max = 255

# # Sobel x on Gray and threshold x gradient and plot
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) 
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= x_thresh_min) & (scaled_sobel <= x_thresh_max)] = 1
ax1.set_title('sxbinary')
ax1.imshow(sxbinary)


# Sobel x on S Channel and threshold x gradient and plot
satsobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) 
satabs_sobelx = np.absolute(satsobelx)
satscaled_sobel = np.uint8(255*satabs_sobelx/np.max(satabs_sobelx))
satsxbinary = np.zeros_like(satscaled_sobel)
satsxbinary[(satscaled_sobel >= x_thresh_min) & (satscaled_sobel <= x_thresh_max)] = 1
ax2.set_title('satsxbinary')
ax2.imshow(satsxbinary)
# Sobel x on R Channel and threshold x gradient and plot
redsobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0)
redabs_sobelx = np.absolute(redsobelx)
redscaled_sobel = np.uint8(255*redabs_sobelx/np.max(redabs_sobelx))
redsxbinary = np.zeros_like(redscaled_sobel)
redsxbinary[(redscaled_sobel >= x_thresh_min) & (redscaled_sobel <= x_thresh_max)] = 1
# ax3.set_title('redsxbinary')
# ax3.imshow(redsxbinary)
# hsv_binary = cv2.inRange(hsv, lower_white, upper_white)
# ax3.set_title('hsv_binary')
# ax3.imshow(hsv_binary)

# hsv_binary2 = cv2.inRange(hsv, lower_white2, upper_white2)
# ax6.set_title('hsv_binary2')
# ax6.imshow(hsv_binary2)

#ADAPTIVE MEAN x on sat
adaptive_blur = cv2.GaussianBlur(s_channel,(5,5),0)
adaptive_mean = cv2.adaptiveThreshold(adaptive_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,5)
ax3.set_title('ADAPTIVE MEAN')
ax3.imshow(adaptive_mean)

#ADAPTIVE GAUS Sobel x on sat
adaptive_gaus = cv2.adaptiveThreshold(adaptive_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,5)
ax6.set_title('ADAPTIVE GAUSSIAN')
ax6.imshow(adaptive_gaus)

# sobelx on adaptive gaus
adaptive_sobelx = cv2.Sobel(adaptive_gaus, cv2.CV_64F,dx = 1, dy = 0) 
adaptiveabs_sobelx = np.absolute(adaptive_sobelx)
adaptive_scaled_sobel = np.uint8(255*adaptiveabs_sobelx/np.max(adaptiveabs_sobelx))
gausbinary = np.zeros_like(adaptive_scaled_sobel)
gausbinary[(adaptive_scaled_sobel >= x_thresh_min) & (adaptive_scaled_sobel <= x_thresh_max)] = 1
ax4.set_title('adaptive_sobelx')
ax4.imshow(adaptive_scaled_sobel)

# # Threshold S Channel and plot
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
# ax4.set_title('s_binary')
# ax4.imshow(s_binary)

# # Threshold HLS space and plot
# hls_binary = cv2.inRange(hls, lower_yellow, upper_yellow)
# ax4.set_title('hls_binary')
# ax4.imshow(hls_binary)

# Stack thresholded sobel x on gray and thresholded saturation and plot
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
ax5.set_title('color_binary')
ax5.imshow(color_binary)

# Combine the two binary thresholds and plot
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
# ax6.set_title('combined_binary')
# ax6.imshow(combined_binary)

# Sobel X on sat thresholded image and plot
sobelx_sat_thresh = cv2.Sobel(s_binary, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx_sat_thresh = np.absolute(sobelx_sat_thresh) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobelx_sat_thresh = np.uint8(255*abs_sobelx_sat_thresh/np.max(abs_sobelx_sat_thresh))
ax7.set_title('scaled_sobelx_sat_thresh')
ax7.imshow(scaled_sobelx_sat_thresh)

# ANDed sxbinary w/ scaled_sobelx_sat_thresh
combined_binary2 = np.zeros_like(sxbinary)
combined_binary2[(scaled_sobelx_sat_thresh > 0) & (sxbinary == 1)] = 1
ax8.set_title('combined_binary2')
ax8.imshow(combined_binary2)

# Original Image
ax9.set_title('original')
ax9.imshow(img)

yellow = np.uint8([[[255,255,0]]])
white = np.uint8([[[255,255,255]]])
hls_yellow = cv2.cvtColor(yellow,cv2.COLOR_RGB2HLS)
hls_white = cv2.cvtColor(white,cv2.COLOR_RGB2HLS)
hsv_white = cv2.cvtColor(white, cv2.COLOR_RGB2HSV)
print( 'hls_yellow = ' )
print( hls_yellow)
print( 'hls white = ')
print(hls_white)
print( 'hsv white = ')
print(hsv_white)

plt.show()