__author__ = 'quale'

import os
import cv2
import numpy as np
'''
    This script is to transform the images in the folder. Using edge detection, then contour detection to find the
    appropriate points to crop the image (rectnagle). Then, a perspective change is done to make the image parallel and
    not skewed.

    http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    TAVI: blur(19,23), canny(80,200)
    M2U00204_PER20F: blur(11,13),  canny(30, 200)
    M2U00207_PER20F: blur(5,3), canny(100,200) -- use find_contours_2
    M2U00236_PER20F: blur(5,3), canny(100,200), epsilon=0.003 -- use find_contours_2
'''

def find_contours(image, blur_params, canny_params):
    orig = image.copy()

    # convert image to gray scale, Gaussian blur, then edge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,3), 0)
    edges = cv2.Canny(blur, 100, 200)

    # close up any open/fragments of the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # find the contours
    contours, hierarchy= cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours,key=cv2.contourArea, reverse=True)[:10]

    total_cnts = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)

        if len(approx) == 4:
            total_cnts.extend(approx)
            #cv2.drawContours(image, approx, -1, (0, 255,0), 3)

    # manual labor to find the right 4 points for each folder
    ######## M2U00204_PER20F ##########
    # total_cnts = total_cnts[8:]
    # print total_cnts.pop(4)
    # print total_cnts.pop(3)
    # print total_cnts.pop(3)
    # print total_cnts.pop(3)
    ####################################

    ########### TAVI_PER1F ##########
    total_cnts = total_cnts[0:len(total_cnts)-2]
    total_cnts.pop(2)
    total_cnts.pop(1)
    print len(total_cnts)
    # cv2.drawContours(image, total_cnts, -1, (0, 255,0), 3)
    # paint(image)
    # return

    return orig, total_cnts

# used for  M2U00207_PER20F
def find_contours_2(image, blur_params, canny_params):
    orig = image.copy()

    # convert image to gray scale, Gaussian blur, then edge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,7), 0)
    edges = cv2.Canny(blur, 100, 200)

    # close up any open/fragments of the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # find the contours
    contours, hierarchy= cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours,key=cv2.contourArea, reverse=True)

    # cv2.drawContours(image, cnts, -1, (0, 255,0), 3)
    # paint(image)
    # return

    total_cnts = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        #approx = cv2.approxPolyDP(c, 0.03*peri, True) # f0r M2U00207_PER20F
        approx = cv2.approxPolyDP(c, 0.003*peri, True) # for M2U00236_PER20F and M2U00241_PER20F

        #if len(approx) > 5 and len(approx) < 7: #top left corner is tricky, reduced to 1, for M2U00207_PER20F
        #if len(approx) > 16 and len(approx) < 18: # for M2U00236_PER20F
        #if len(approx) > 13 and len(approx) < 15: # for M2U00241_PER20F
        if 13 < len(approx) < 16 :
            total_cnts.extend(approx)
            #cv2.drawContours(image, approx, -1, (0, 255,0), 3)

    # manual labor to find the right 4 points for each folder
    ################# M2U00207_PER20F ##########
    # total_cnts = total_cnts[-12:] # first get last 12 items (know it contains 4 points of wanted screen)
    # total_cnts = total_cnts[:5] # get the first five
    # total_cnts.pop(2) # remove the 2nd

    ############## M2U00236_PER20F ################
    #total_cnts = total_cnts[-22:]
    #total_cnts = total_cnts[0:4]

    ############## M2U00241_PER20 F################
    total_cnts = total_cnts[-46:]
    total_cnts = total_cnts[:-23]
    found = total_cnts[:3]
    total_cnts = total_cnts[22:len(total_cnts)]
    total_cnts.extend(found)


    # print len(total_cnts)
    # cv2.drawContours(image, total_cnts, -1, (0, 255,0), 3)
    # paint(image)
    # return

    return orig, total_cnts

def perspective_transform(orig, points, image_name):
    pts = np.asarray(points)
    pts = pts.squeeze() # shape (4,2)
    rect = np.zeros(pts.shape, dtype='float32')

    s = pts.sum(axis=1) # sum along the row
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # compute width of rectangle for new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    #  compute the height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct the destination points which will be used to
    # map the screen to a top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # change directory to save new warped images into
    #path = '/Users/quale/Desktop/CatheterWarps/M2U00204_PER20F'
    #path = '/Users/quale/Desktop/CatheterWarps/M2U00207_PER20F'
    #path = '/Users/quale/Desktop/CatheterWarps/M2U00236_PER20F'
    path = '/Users/quale/Desktop/CatheterWarps/M2U00241_PER20F'
    #path ='/Users/quale/Desktop/CatheterWarps/TAVI_PER1F'
    os.chdir(path)
    cv2.imwrite(image_name,warp)


def paint(img):
    cv2.imshow('new image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    path = '/Users/quale/Desktop/CatheterLabels/'
    folders = []
    for filename in os.listdir(path):
        folders.append(filename)

    # get the last 5 folders which needs to be transformed
    folders = folders[7:len(folders)]
    for f in folders:
        folder_path = path + '/' + f
        print 'working on..', folder_path
        images = os.listdir(folder_path)

        it = 0
        for image in images:
            if 'frame' in image:
                print image
                p = folder_path + '/'+ image
                org_image = cv2.imread(p)
                if it == 0:
                    #org, corner_pts = find_contours(org_image, (19,23), (80,200))
                    org, corner_pts = find_contours_2(org_image, (19,23), (80,200))
                    perspective_transform(org, corner_pts, image)
                else:
                    perspective_transform(org_image, corner_pts, image)
                it+=1
        break #for debugging

if __name__ == '__main__':
    main()