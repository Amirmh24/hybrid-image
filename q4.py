import numpy as np
import cv2
import math


def redEye(img, eyeL, eyeR):
    imgRedEye = img
    a = 2
    imgRedEye[eyeL[0] - a:eyeL[0] + a, eyeL[1] - a:eyeL[1] + a, :] = [0, 0, 255]
    imgRedEye[eyeR[0] - a:eyeR[0] + a, eyeR[1] - a:eyeR[1] + a, :] = [0, 0, 255]
    return imgRedEye


def distance(p1, p2):
    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)
    return dist


def rotate(img, eyeL, eyeR):
    imgRotated = np.zeros(img.shape)
    hei, wid, chan = imgRotated.shape
    centeri, centerj = int(wid / 2), int(hei / 2)
    center = (centeri, centerj)
    theta = math.atan((eyeR[0] - eyeL[0]) / (eyeR[1] - eyeL[1]))
    matrixRotation = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    eyeL = (np.matmul(matrixRotation, [(eyeL[0] - centerj), (eyeL[1] - centeri)]) + [centerj, centeri]).astype(int)
    eyeR = (np.matmul(matrixRotation, [(eyeR[0] - centerj), (eyeR[1] - centeri)]) + [centerj, centeri]).astype(int)
    theta = theta * 180 / math.pi
    matrixRotation = cv2.getRotationMatrix2D(center, theta, 1.0)
    imgRotated = cv2.warpAffine(img, matrixRotation, (wid, hei))
    return imgRotated, eyeL, eyeR


def scale(img1, img2, eyeL1, eyeR1, eyeL2, eyeR2):
    dist1 = distance(eyeL1, eyeR1)
    dist2 = distance(eyeL2, eyeR2)
    firstImageIsLarger = True
    imgLarge, eyeLLarge, eyeRLarge = img1, eyeL1, eyeR1
    imgSmall, eyeLSmall, eyeRSmall = img2, eyeL2, eyeR2
    if (dist1 <= dist2):
        imgLarge, eyeLLarge, eyeRLarge = img2, eyeL2, eyeR2
        imgSmall, eyeLSmall, eyeRSmall = img1, eyeL1, eyeR1
        firstImageIsLarger = False
    heiLarge, widLarge, channels = imgLarge.shape
    distLarge = distance(eyeLLarge, eyeRLarge)
    distSmall = distance(eyeLSmall, eyeRSmall)
    rr = distLarge / distSmall
    imgLarge = cv2.resize(imgLarge, (int(widLarge / rr), int(heiLarge / rr)))
    eyeLLarge[0], eyeLLarge[1] = int(eyeLLarge[0] / rr), int(eyeLLarge[1] / rr)
    eyeRLarge[0], eyeRLarge[1] = int(eyeRLarge[0] / rr), int(eyeRLarge[1] / rr)
    if (firstImageIsLarger):
        return imgLarge, imgSmall, eyeLLarge, eyeRLarge, eyeLSmall, eyeRSmall
    else:
        return imgSmall, imgLarge, eyeLSmall, eyeRSmall, eyeLLarge, eyeRLarge


def match(img1, img2, eyeL1, eyeR1, eyeL2, eyeR2):
    dx = eyeL1[1] - eyeL2[1]
    dy = eyeL1[0] - eyeL2[0]
    matrixTransport = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    img2 = cv2.warpAffine(img2, matrixTransport, (img1.shape[1], img1.shape[0]))
    eyeL2[0] = eyeL2[0] + dy
    eyeL2[1] = eyeL2[1] + dx
    eyeR2[0] = eyeR2[0] + dy
    eyeR2[1] = eyeR2[1] + dx
    return img1, img2, eyeL1, eyeR1, eyeL2, eyeR2


def getFaces(img1, img2, eyeL1, eyeR1, eyeL2, eyeR2):
    img1, eyeL1, eyeR1 = rotate(img1, eyeL1, eyeR1)
    img2, eyeL2, eyeR2 = rotate(img2, eyeL2, eyeR2)
    img1, img2, eyeL1, eyeR1, eyeL2, eyeR2 = scale(img1, img2, eyeL1, eyeR1, eyeL2, eyeR2)
    img1, img2, eyeL1, eyeR1, eyeL2, eyeR2 = match(img1, img2, eyeL1, eyeR1, eyeL2, eyeR2)
    centeri, centerj = int((eyeL1[0] + eyeR1[0]) / 2), int((eyeL1[1] + eyeR1[1]) / 2)
    dist = distance(eyeL1, eyeR1)
    img1 = img1[int(centeri - 2 * dist):int(centeri + 2 * dist), int(centerj - 1.5 * dist):int(centerj + 1.5 * dist), :]
    img2 = img2[int(centeri - 2 * dist):int(centeri + 2 * dist), int(centerj - 1.5 * dist):int(centerj + 1.5 * dist), :]
    return img1, img2


def dftFilter(img):
    DFT = np.zeros(img.shape, dtype='complex_')
    hei, wid, chan = DFT.shape
    for ch in range(chan):
        DFT[:, :, ch] = np.fft.fftshift(np.fft.fft2(img[:, :, ch]))
    return DFT


def idftFilter(img):
    iDFT = np.zeros(img.shape, dtype='complex_')
    hei, wid, chan = iDFT.shape
    for ch in range(chan):
        iDFT[:, :, ch] = np.fft.ifft2(np.fft.ifftshift(img[:, :, ch]))
    return iDFT


def gaussFilter(img, sigma, type):
    gaussMatrix = np.zeros(img.shape)
    hei, wid, chan = gaussMatrix.shape
    i0 = int(hei / 2)
    j0 = int(wid / 2)
    for i in range(hei):
        for j in range(wid):
            if (type == "highpass"):
                gaussMatrix[i, j, :] = 1 - np.exp(((i - i0) ** 2 + (j - j0) ** 2) / (-2 * sigma ** 2))
            if (type == "lowpass"):
                gaussMatrix[i, j, :] = np.exp(((i - i0) ** 2 + (j - j0) ** 2) / (-2 * sigma ** 2))
    return gaussMatrix


def cutoffFilter(img, threshhold, type):
    cutoffMatrix = np.zeros(img.shape)
    hei, wid, chan = cutoffMatrix.shape
    i0 = int(hei / 2)
    j0 = int(wid / 2)
    for i in range(hei):
        for j in range(wid):
            if (type == "highpass"):
                if (distance([i, j], [i0, j0]) > threshhold):
                    cutoffMatrix[i, j, :] = 1
                else:
                    cutoffMatrix[i, j, :] = 0
            if (type == "lowpass"):
                if (distance([i, j], [i0, j0]) < threshhold):
                    cutoffMatrix[i, j, :] = 1
                else:
                    cutoffMatrix[i, j, :] = 0
    return cutoffMatrix


def normalize(img):
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img


# joe
I1 = cv2.imread("q4_01_near.jpg")
eyeL1, eyeR1 = [257, 288], [278, 386]
# trump
I2 = cv2.imread("q4_02_far.jpg")
eyeL2, eyeR2 = [348,330], [345,449]


I1, I2 = getFaces(I1, I2, eyeL1, eyeR1, eyeL2, eyeR2)
cv2.imwrite("q4_03_near.jpg", I1)
cv2.imwrite("q4_04_far.jpg", I2)

height, width, channels = I1.shape
dft1 = dftFilter(I1)
dft2 = dftFilter(I2)

cv2.imwrite("q4_05_dft_near.jpg", normalize(np.log10(np.absolute(dft1 + 1))))
cv2.imwrite("q4_06_dft_far.jpg", normalize(np.log10(np.absolute(dft2 + 1))))

r = 15
s = 10

gauss1 = gaussFilter(I1, r, "highpass")
gauss2 = gaussFilter(I2, s, "lowpass")

cv2.imwrite("q4_07_highpass_" + str(r) + ".jpg", normalize(gauss1))
cv2.imwrite("q4_08_lowpass_" + str(s) + ".jpg", normalize(gauss2))

cutoff1 = cutoffFilter(I1, 10, "highpass")
cutoff2 = cutoffFilter(I2, 20, "lowpass")

filter1 = cutoff1 * gauss1
filter2 = cutoff2 * gauss2

cv2.imwrite("q4_09_highpass_cutoff.jpg", normalize(filter1))
cv2.imwrite("q4_10_lowpass_cutoff.jpg", normalize(filter2))

dftFiltered1 = dft1 * filter1
dftFiltered2 = dft2 * filter2

cv2.imwrite("q4_11_highpassed.jpg", normalize(np.log10(np.absolute(dftFiltered1) + 1)))
cv2.imwrite("q4_12_lowpassed.jpg", normalize(np.log10(np.absolute(dftFiltered2) + 1)))

hybridFrequency = np.zeros(dftFiltered2.shape, dtype='complex_')
hybridFrequency[dftFiltered2 == 0] = dftFiltered1[dftFiltered2 == 0]
hybridFrequency[dftFiltered1 == 0] = dftFiltered2[dftFiltered1 == 0]
hybridFrequency[(dftFiltered1 != 0) & (dftFiltered2 != 0)] = (dftFiltered1[(dftFiltered1 != 0) & (dftFiltered2 != 0)] +
                                                              dftFiltered2[
                                                                  (dftFiltered1 != 0) & (dftFiltered2 != 0)]) / 2

cv2.imwrite("q4_13_hybrid_frequency.jpg", normalize(np.log10(np.absolute(hybridFrequency) + 1)))
hybrid = np.real(idftFilter(hybridFrequency))
cv2.imwrite("q4_14_hybrid_near.jpg", hybrid)
hybrid = cv2.blur(hybrid, (5, 5))
hybrid = cv2.resize(hybrid, (int(hybrid.shape[1] / 4), int(hybrid.shape[0] / 4)))
cv2.imwrite("q4_15_hybrid_far.jpg", hybrid)
