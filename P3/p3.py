import numpy as np
from PIL import Image

from scipy import ndimage, signal

############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    img = Image.open(filename).convert('RGB')
    img = np.array(img).astype(float)
    img = img / 255.
    return img

### TODO 2: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    f = np.zeros((k,k))
    center = k//2
    for i in range(k):
        for j in range(k):
            f[i,j] = np.exp(-(float(i-center)**2 + float(j-center)**2)/(2*sigma*sigma))
    f = f/np.sum(f)
    return f

### TODO 3: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. (use scipy.signal.convolve)
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel and convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel. (use scipy.signal.convolve) 
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    img = 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
    img = signal.convolve(img, gaussian_filter(5,1), mode='same')
    gx = signal.convolve(img, np.array([[0.5, 0, -0.5]]), mode='same')
    gy = signal.convolve(img, np.array([[0.5],[0],[-0.5]]), mode='same')
    mag = np.sqrt(gx*gx + gy*gy)
    ori = np.arctan2(gy, gx)
    return mag, ori

##########----------------Line detection----------------

### TODO 4: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are arrays representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    return (np.abs(x*np.cos(theta) + y*np.sin(theta) + c)<thresh)

### TODO 5: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. Each line must appear as red on the final image
### where every pixel which is less than thresh units away from the line should be colored red
def draw_lines(img, lines, thresh):
    img = img.copy()
    I, J = np.unravel_index(np.arange(img.shape[0]*img.shape[1]), (img.shape[0], img.shape[1]))
    for (theta, c) in lines:
        idx = np.where(check_distance_from_line(J, I, theta, c, thresh))[0]
        img[I[idx], J[idx],0] = 1
        img[I[idx], J[idx],1] = 0
        img[I[idx], J[idx],2] = 0
    return img
 
### TODO 6: Do Hough voting. You get as input the gradient magnitude and the gradient orientation, as well as a set of possible theta values and a set of possible c
### values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1
### (b) Its distance from the (theta, c) line is less than thresh2, and
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    bmap = gradmag>thresh1
    I, J = np.where(bmap)
    votes = np.zeros((len(thetas), len(cs)))
    for i in range(len(thetas)):
        for j in range(len(cs)):
            theta = thetas[i]
            c = cs[j]
            val = check_distance_from_line(J, I, theta, c, thresh2)
            ori = np.abs(gradori[I,J] - theta)
            votes[i,j] = votes[i,j] + np.sum(val & (ori<thresh3))
    return votes
    
### TODO 7: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if (a) its votes are greater than thresh, and 
### (b) its value is the maximum in a (nbhd x nbhd) neighborhood in the votes array.
### Return a list of (theta, c) pairs
def localmax(votes, thetas, cs, thresh,nbhd):
    chosenI = []
    chosenJ = []
    k=nbhd//2
    for i in range(0,votes.shape[0]):
        for j in range(0, votes.shape[1]):
            lowi = max(0, i-k)
            lowj = max(0,j-k)
            highi = min(votes.shape[0], i+k+1)
            highj = min(votes.shape[1], j+k+1)
            if (votes[i,j]>thresh) and (votes[i,j] >= np.max(votes[lowi:highi,lowj:highj])):
                chosenI.append(i)
                chosenJ.append(j)
    chosenI = np.array(chosenI)
    chosenJ = np.array(chosenJ)
    return list(zip(thetas[chosenI], cs[chosenJ]))

# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori,thetas, cs, 0.2, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
