import dlib
import cv2
import numpy as np
import math

predictor_path = 'D:/dlib-shape/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        # for idx,point in enumerate(land_marks_node):
        #     # 68点坐标
        #     pos = (point[0,0],point[0,1])
        #     print(idx,pos)
        #     # 利用cv2.circle给每个特征点画一个圈，共68个
        #     cv2.circle(img_src, pos, 5, color=(0, 255, 0))
        #     # 利用cv2.putText输出1-68
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(land_marks_node)

    return land_marks


def getEllipseCross(p1x, p1y, p2x, p2y, a, b, centerX, centerY):
    resx = 0
    resy = 0
    k = (p1y - p2y) / (p1x - p2x);
    m = p1y - k * p1x;
    A = (b * b + (a * a * k * k))
    B = 2 * a * a * k * m
    C = a * a * (m * m - b * b)

    X1 = (-B + math.sqrt(B * B - (4 * A * C))) / (2 * A)
    X2 = (-B - math.sqrt(B * B - (4 * A * C))) / (2 * A)

    # Y1 = math.sqrt(1 - (b * b * X1 * X1 ) / (a * a) )
    # Y2 = math.sqrt(1 - (b * b * X2 * X2 ) / (a * a) )

    Y1 = k * X1 + m
    Y2 = k * X2 + m

    if getDis(p2x, p2y, X1, Y1) < getDis(p2x, p2y, X2, Y2):
        resx = X1
        resy = Y1
    else:
        resx = X2
        resy = Y2

    return [resx + centerX, resy + centerY]


def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]

def getDis(p1x, p1y, p2x, p2y):
    return math.sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y))

def get_line_cross_point(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    # print(p1x, p1y)
    # print(p2x, p2y)
    # print(p3x, p3y)
    # print(p4x, p4y)
    a0, b0, c0 = getLinearEquation(p1x, p1y, p2x, p2y)
    a1, b1, c1 = getLinearEquation(p3x, p3y, p4x, p4y)
    # print(a0,b0,c0)
    # print(a1,b1,c1)

    D = a0*b1-a1*b0
    if D==0:
        return None
    x = (b0*c1-b1*c0)/D
    y = (a1*c0-a0*c1)/D
    return x, y

def localTranslationWarp(srcImg, startIndex, endIndex,Strength,landmarks_node):

    midIndex = (startIndex + endIndex + 1) >> 1

    startDot = landmarks_node[startIndex]
    endDot = landmarks_node[endIndex]
    midDot = landmarks_node[midIndex]

    Eye = []
    for i in range(startIndex, endIndex+1):
        Eye.append([landmarks_node[i][0, 0], landmarks_node[i][0, 1]])
    ellipseEye = cv2.fitEllipse(np.array(Eye))
    # cv2.ellipse(srcImg, ellipseEye, (0, 255, 0), 1)
    # cv2.imshow("eli",srcImg)


    radius = math.sqrt(
        (startDot[0, 0] - midDot[0, 0]) * (startDot[0, 0] - midDot[0, 0]) -
        (startDot[0, 1] - midDot[0, 1]) * (startDot[0, 1] - midDot[0, 1])
    ) / 2
    list = []

    for i in range(0,3):

        tmplist = []
        tmplist = get_line_cross_point(
                        landmarks_node[startIndex + i][0, 0], landmarks_node[startIndex + i][0, 1],
                        landmarks_node[midIndex + i][0, 0], landmarks_node[midIndex + i][0, 1],
                        landmarks_node[startIndex + ((i + 1) % 3)][0, 0], landmarks_node[startIndex + ((i + 1) % 3)][0, 1],
                        landmarks_node[midIndex + ((i + 1) % 3)][0, 0], landmarks_node[midIndex + ((i + 1) % 3)][0, 1]
                  )
        list.append(tmplist)
    # for l in list:
    #     print(l)

    a = getDis(list[0][0], list[0][1], list[1][0], list[1][1])
    b = getDis(list[1][0], list[1][1], list[2][0], list[2][1])
    c = getDis(list[2][0], list[2][1], list[0][0], list[0][1])

    centerX = (a * list[0][0] + b * list[1][0] + c * list[2][0]) / (a + b + c)
    centerY = (a * list[0][1] + b * list[1][1] + c * list[2][1]) / (a + b + c)
    # print(centerX)
    # print(centerY)
    # print(" ")
    width, height, cou = srcImg.shape
    Intensity = 15*512*512/(width*height)

    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    K0 = Strength / 100.0

    # 计算公式中的|m-c|^2

    eyeWidth = radius
    eyeHeight = getDis((landmarks_node[startIndex+1][0, 0] + landmarks_node[startIndex+2][0, 0]) / 2,
                       (landmarks_node[startIndex+1][0, 1] + landmarks_node[startIndex+2][0, 1]) / 2,
                       (landmarks_node[midIndex+1][0, 0] + landmarks_node[midIndex+2][0, 0]) / 2,
                       (landmarks_node[midIndex+1][0, 1] + landmarks_node[midIndex+2][0, 1]) / 2)
    centerX = ellipseEye[0][0]
    centerY = ellipseEye[0][1]
    ellipseA = ellipseEye[1][1]
    ellipseB = ellipseEye[1][0]
    ellipseC = math.sqrt(ellipseA * ellipseA - ellipseB * ellipseB)
    # print(ellipseA, ellipseB, ellipseC)
    # print(centerX, centerY)
    # ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    #
    for i in range(width):
        for j in range(height):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中

            # if math.fabs(i - centerX) > ((eyeHeight / 2) * 1.5) or math.fabs(j - centerY) > ((eyeWidth / 2) * 1.5):
            #     continue




            if getDis(i, j, centerX - ellipseC, centerY) + getDis(i, j, centerX + ellipseC, centerY) > 2 * ellipseA:
                continue
#             print(i, j)
            [crossX, crossY] = getEllipseCross(0, 0, i - ellipseEye[0][0], j - ellipseEye[0][1], ellipseEye[1][1],
                                               ellipseEye[1][0], ellipseEye[0][0], ellipseEye[0][1])

#             print(crossX, crossY)

            radius = getDis(centerX, centerY, crossX, crossY)
            ddradius = radius * radius
            distance = (i - centerX) * (i - centerX) + (j - centerY) * (j - centerY)
            K1 = 1.0 - (1.0 - distance / ddradius) * K0


            # 映射原位置
            UX = (i - centerX) * K1 + centerX
            UY = (j - centerY) * K1 + centerY
#             print(UX, UY)

            # 根据双线性插值法得到UX，UY的值
            value = BilinearInsert(srcImg, UX, UY)
            # 改变当前 i ，j的值
            copyImg[j, i] = value

    return copyImg


# 双线性插值法
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


def face_thin_auto(src,LStrength,RStrength):
    landmarks = landmark_dec_dlib_fun(src)

    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        # print(landmarks_node)
        bigEyeImage = localTranslationWarp(src,36,41,LStrength,landmarks_node)
        bigEyeImage = localTranslationWarp(bigEyeImage,42,47,RStrength,landmarks_node)



#     cv2.imshow('bigEye', bigEyeImage)
    cv2.imwrite('C:/Users/mibbp/Pictures/bigEye.jpg', bigEyeImage)


def test():
    print('bigEyeTest')

def main(LStrength, RStrength):
    LStrength = int(LStrength)
    RStrength = int(RStrength)
    src = cv2.imread('C:/Users/mibbp/Pictures/bytest.jpg')
#     cv2.imshow('src', src)
    face_thin_auto(src,LStrength,RStrength)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()