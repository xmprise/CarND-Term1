import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
# Image library used for extracting test images


# Need to see if the color of the road surface is converted to a gray scale and the intensity differs.
def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Perform blur operation to remove noise. I will use Gaussian here.
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# Use edge detection to extract lines from the frame
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


# Applies an image mask, only keeps the region of the image defined by the polygon
# formed from `vertices`. The rest of the image is set to black.
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


# Determines a straight line in Hough space and visualizes the returned coordinates as a straight line
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    right_segment = []
    left_segment = []
    top_y = 320
    bottom_y = img.shape[0]

    # Divide the segment to the right and left using the sign of the slope
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                slope = -1
            else:
                slope = ((y2 - y1) / (x2 - x1))

            if slope >= 0:
                right_segment.append([x1, y1])
                right_segment.append([x2, y2])

            elif slope < 0:
                left_segment.append([x1, y1])
                left_segment.append([x2, y2])

    # Connect the coordinates stored in the segment list with a straight line.
    # Use the fixed y-coordinate and simple straight-line equation to find and connect the x-coordinates.
    if len(right_segment) > 0:
        right_segment = np.array(right_segment)
        right_top_x, right_bottom_x = coordinate_x(right_segment, top_y, bottom_y)
        cv2.line(img, (right_top_x, top_y), (right_bottom_x, bottom_y), color, thickness)

    if len(left_segment) > 0:
        left_segment = np.array(left_segment)
        left_top_x, left_bottom_x = coordinate_x(left_segment, top_y, bottom_y)
        cv2.line(img, (left_top_x, top_y), (left_bottom_x, bottom_y), color, thickness)


def coordinate_x(segment, top_y, bottom_y):
    [x1, y1, x2, y2] = cv2.fitLine(segment, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)

    top_x = (top_y - (y2 - ((y1 / x1) * x2))) / (y1 / x1)
    bottom_x = (bottom_y - (y2 - ((y1 / x1) * x2))) / (y1 / x1)

    return top_x, bottom_x


# The coordinates of the obtained straight line are converted into coordinates to be visualized
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def process_image(image):
    gray = gray_scale(image)

    # Reduce the kernel size from 5 to 3, as suggested by the reviewer
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Assign masking range to lane shape
    vertices = np.array([[(60, image.shape[0]), (450, 320), (900, 540), (540, 320)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    rho = 1
    theta = np.pi / 180
    # Increase threshold of hough transform according to suggestion of reviewer
    threshold = 20
    # Increase min_line_len and max_line_gap for Hough Transform to make your lines longer with less number of breaks
    min_line_length = 40
    max_line_gap = 30
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    weighted_image = weighted_img(image, lines)

    return weighted_image


output = 'output.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
out_clip = clip1.fl_image(process_image)
out_clip.write_videofile(output, audio=False)
