import cv2
import numpy as np


def RGB2HSV(src):
    h, w, c = src.shape
    dst = np.empty((h, w, c))

    for y in range(0, h):
        for x in range(0, w):
            [b, g, r] = src[y][x]/255.0
            mx = max(r, g, b)
            mn = min(r, g, b)

            diff = mx - mn

                        # Hの値を計算
            if mx == mn : h = 0
            elif mx == r : h = 60 * ((g-b)/diff)
            elif mx == g : h = 60 * ((b-r)/diff) + 120
            elif mx == b : h = 60 * ((r-g)/diff) + 240
            if h < 0 : h = h + 360

            # Sの値を計算
            if mx != 0:s = diff/mx
            else: s = 0

            # Vの値を計算
            v = mx

            # Hを0～179, SとVを0～255の範囲の値に変換
            dst[y][x] = [h * 0.5, s * 255, v * 255]

img1 = cv2.imread("001.jpg")
img1 = cv2.resize(img1, (1028, 1028));

hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([10, 0, 0])
upper_yellow = np.array([60, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
out = cv2.bitwise_and(img1, img1, mask= mask)

# 膨張収縮でノイズ除去
kernel = np.ones((5,5),np.uint8)
noiseless = mask
noiseless = cv2.dilate(noiseless, kernel)
noiseless = cv2.dilate(noiseless, kernel)
noiseless = cv2.erode(noiseless, kernel)
noiseless = cv2.erode(noiseless, kernel)
noiseless = cv2.erode(noiseless, kernel)
noiseless = cv2.erode(noiseless, kernel)
noiseless = cv2.dilate(noiseless, kernel)
noiseless = cv2.dilate(noiseless, kernel)


# エッジ抽出
edge = noiseless
edge = cv2.Laplacian(edge, cv2.CV_8U)

# ハフ変換

lines = cv2.HoughLines(edge, rho= 1, theta= np.pi/180, threshold= 80);

# if lines != None:
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img1, (x1,y1), (x2,y2), (0,0,255), 2)



cv2.imshow("out", img1)

cv2.waitKey(0)






