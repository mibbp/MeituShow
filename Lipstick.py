# encoding:utf-8
from collections import deque

import dlib
import numpy as np
import cv2


# 4/18 cy:

# 将像素遍历范围缩小至嘴唇所在的矩形
#
# 下一步：探究bfs（魔棒法）在边缘部分的可行性？


def rect_to_bb(rect):  # [psy]:获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def shape_to_np(shape, dtype="int"):  # [psy]:将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def resize(image, width=600):  # [psy]:将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def getbgr(image, xx, yy):
    bb = 0
    gg = 0
    rr = 0
    bgr = image[yy, xx]
    for ii in range(-1, 1):
        for jj in range(-1, 1):
            if ii == 0 and jj == 0:
                continue
            tx = xx + ii
            ty = yy + jj
            tbgr = image[ty, tx]
            bb += tbgr[0]
            gg += tbgr[1]
            rr += tbgr[2]

    bb = int(bb / 8 / 3 + bgr[0] / 3 * 2)
    gg = int(gg / 8 / 3 + bgr[1] / 3 * 2)
    rr = int(rr / 8 / 3 + bgr[2] / 3 * 2)
    print(bb, gg, rr)
    return bgr


def bfs(image, edge_list, convex):
    que = deque()
    dis = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    for node in edge_list:
        que.append((node[0], node[1]))
        dis[node[0], node[1]] = -10

    cc = 0
    while que:
        cc += 1
        now = que.popleft()
        # nbgr = image[now[1], now[0]]
        if dis[now[0], now[1]] + 1 < 0:
            for xx in range(-1, 1):
                for yy in range(-1, 1):
                    ty = yy + now[1]
                    tx = xx + now[0]
                    if cv2.pointPolygonTest(convex, (tx, ty), False) < 0:
                        if dis[tx, ty] > dis[now[0], now[1]] + 1:
                            dis[tx, ty] = dis[now[0], now[1]] + 1
                            tbgr = image[ty, tx]
                            if tbgr[0] + 40 < 255:
                                image[ty, tx][0] = tbgr[0] + 40
                            if tbgr[1] - 10 < 255:
                                image[ty, tx][1] = tbgr[1] - 10
                            if tbgr[2] - 10 < 255:
                                image[ty, tx][2] = tbgr[2] - 10
                            if (tx, ty) not in que:
                                que.append((tx, ty))
    print(cc)
    return image


def outside_edge_judge(x, y, d1, d2, edgelist, threshold, img):  # [cy]:边缘点判断函数
    a1 = 1 if -threshold <= d1 <= threshold else 0
    a2 = 1 if -threshold <= d2 <= threshold else 0
    if a1 + a2 == 1:
        edgelist.append([y, x])
        # img[y, x] = [0, 0, 255]
        return True
    return False


def inside_edge_judge(x, y, d1, edgelist, threshold, img):
    a = 1 if -threshold <= d1 <= threshold else 0
    if a == 1:
        edgelist.append([y, x])
        return True
    return False


def mouth_area_rect(points):  # [cy]:获取嘴唇像素的矩形
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins[0], maxs[0] + 1, mins[1], maxs[1] + 1


def most_frequently_hsv(img, array):  # [cy]:获取嘴唇内部像素出现最多的点坐标
    arr = []
    for pixel in array:
        yy = pixel[0]
        xx = pixel[1]
        ph = img[yy, xx][0]
        ps = img[yy, xx][1]
        pv = img[yy, xx][2]
        arr.append([ph, ps, pv])
    arr = np.array(arr)
    h = np.argmax(np.bincount(arr[:, 0]))
    s = np.argmax(np.bincount(arr[:, 1]))
    v = np.argmax(np.bincount(arr[:, 2]))
    return int(h), int(s), int(v)


def color_delta(c1, c2):
    c1 = np.array(c1)
    c2 = np.array(c2)
    delta = c1 - c2
    return np.max(delta) - np.min(delta)


def exact_lip_points(img, all_points, color):  # [cy]:在给定坐标的像素们中筛选出与color相近的像素的坐标们
    color = np.array(color)
    exact_points = []
    delta = 20
    for pixel in all_points:
        pixel_hsv = np.array(img[pixel[0], pixel[1]])  # [cy]:某点的hsv像素值
        if color_delta(color, pixel_hsv) < delta:  # [cy]:与参考色color的色差
            exact_points.append(pixel)

    return exact_points


def side_edge_move(edge_points_array, type, color, min_delta, max_delta, min_e, max_e, img):
    """嘴唇边缘移动

    :param edge_points_array:边缘点
    :param type:类型 'outside' 'inside'
    :param color:参照色
    :param min_delta:逃离时的色差阈值（小于就逃离）
    :param max_delta:靠拢时的色差阈值（大于就靠拢)
    :param min_e:逃离嘴唇的最大距离
    :param max_e:靠拢嘴唇的最大距离
    :param img:图像
    :return:
    """
    res = {}
    rgb_color = (154, 107, 113)
    # color=(rgb_color[0]*180,rgb_color[1]*255,rgb_color[2]*255)
    if type == 'outside':
        type = 1
    elif type == 'inside':
        type = -1
    else:
        type = 0
    # [cy]:首先找出每一列的两个边界点 [0]是上边界，[1]是下边界
    for px in edge_points_array:
        x = px[1]
        y = px[0]
        if x in res.keys():
            if y < res[x][0]:
                res[x][0] = y
            if y > res[x][1]:
                res[x][1] = y
        else:
            res[x] = [y, y]
    # [cy]:确保每列都有上下两个边界
    for x, yrange in res.items():
        if res[x][0] == res[x][1]:
            if x + 1 in res.keys():
                res[x][0] = res[x + 1][0]
                res[x][1] = res[x + 1][1]
            else:
                res[x][0] = res[x - 1][0]
                res[x][1] = res[x - 1][1]
    # [cy]:遍历每个边界
    if type != 0:
        for x, yrange in res.items():
            direction = 0 if type == 1 else 1
            upside_hsv = img[yrange[direction], x]  # 外边缘是上边界，内边缘是下边界
            e = 0
            # 外边缘上边界在嘴唇，则向上逃离
            # 内边缘下边界在嘴唇，也向上逃离
            while color_delta(color, upside_hsv) - min_delta < 0 and e > -min_e:
                e -= 1
                upside_hsv = img[yrange[direction] + e, x]
            # 外边缘上边界在外部，则向下靠拢
            # 内边缘下边界在外部，也向下靠拢
            while color_delta(color, upside_hsv) - max_delta > 0 and e < max_e:
                e += 1
                upside_hsv = img[yrange[direction] + e, x]
            res[x][direction] = yrange[direction] + e

            direction = 1 - direction
            downside_hsv = img[yrange[direction], x]  # 外边缘是下边界，内边缘是上边界
            e = 0
            # 外边缘下边界在嘴唇，则向下逃离
            # 内边缘上边界在嘴唇，也向下逃离
            while color_delta(color, downside_hsv) - min_delta < 0 and e < min_e:
                e += 1
                downside_hsv = img[yrange[direction] + e, x]
            # 外边缘下边界在外部，则向上靠拢
            # 内边缘上边界在外部，也向上靠拢
            while color_delta(color, downside_hsv) - max_delta > 0 and e > -max_e:
                e += -1
                downside_hsv = img[yrange[direction] + e, x]
            res[x][direction] = yrange[direction] + e
    return res


def outline(res, img, color):
    for x, yrange in res.items():
        img[yrange[0], x] = (0, 0, 0)
        img[yrange[1], x] = (0, 0, 255)


def padding_inside_points(pts, edge_points):
    # 0,12为左侧两嘴角
    # 6,16为右侧两嘴角
    left_x1 = pts[0][0]
    left_x2 = pts[12][0]
    y = pts[12][1]
    for x in range(left_x1, left_x2):
        edge_points.append([y, x])
    right_x1 = pts[16][0]
    right_x2 = pts[6][0]
    y = pts[16][1]
    for x in range(right_x1, right_x2):
        edge_points.append([y, x])


def feature(path, color, outside_tuple=(40, 50, 3, 2), inside_tuple=(90, 100, 3, 1)):
    (a1, a2, a3, a4) = outside_tuple
    (b1, b2, b3, b4) = inside_tuple
    detector = dlib.get_frontal_face_detector()  # [cy]:人脸检测仪
    predictor = dlib.shape_predictor("D:/dlib-shape/shape_predictor_68_face_landmarks.dat")  # [cy]:关键点检测器
    image = cv2.imread(path)  # [cy]:读取 输入图像.jpg
    image = resize(image, width=600)  # [cy]:缩放 图像，宽为1200
    print(image.shape)  # [cy]:image尺寸 (高,宽,3)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # [psy]:转为灰度图
    # [cy]:
    # 传入灰度图像，检测出里面的所有脸，某张脸的矩形：rect=[(左上角坐标),(右下角坐标)]
    # OpenCV坐标
    # (0,0) - (100,0)
    #   |        |
    # (0,100) - (100,100)
    rects = detector(gray, 1)  # [psy]:灰度图里定位人脸
    shapes = []  # [psy]:shapes存储找到的人脸框，人脸框仅包含四个角数值如frontal_face_detector.png所示。
    for (i, rect) in enumerate(rects):  # [cy]:遍历所有脸的方框
        shape = predictor(gray, rect)  # [cy]:用关键点检测器检测出关键点们
        shape = shape_to_np(shape)  # [cy]:关键点们变成numpy数组
        shape = shape[48:]  # [cy]:关键点[48-67]是嘴唇区域
        shapes.append(shape)  # [cy]:把这张脸的嘴唇关键点插入到shapes
    # [psy]:
    # 图片转为hsv形式，色调（H），饱和度（S），亮度（V）
    # H:  0 — 180
    # S:  0 — 255
    # V:  0 — 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # [cy]:
    for shape in shapes:
        # [cy]:遍历每个嘴唇
        # [cy]:获取嘴唇像素的矩形
        xstart, xend, ystart, yend = mouth_area_rect(shape)
        # [cy]:图像的高与宽
        # sx, sy = image.shape[0], image.shape[1]
        # [cy]:外嘴唇左半边凸包
        hull_left = cv2.convexHull(np.concatenate((shape[0:4], shape[7:12])))
        # [cy]:外嘴唇右半边凸包
        hull_right = cv2.convexHull(shape[3:8])
        # [psy]:内嘴唇凸包
        hull_inside = cv2.convexHull(shape[12:])
        # [cy]:嘴唇外边缘上的点
        outside_edge_points = []
        # [cy]:嘴唇内边缘上的点
        inside_edge_points = []
        # [cy]:嘴唇上的所有点
        lip_points = []
        for xx in range(xstart, xend):
            for yy in range(ystart, yend):
                # [cy]:获得(xx,yy)到左外凸包的距离，正数说明在内部，measureDist:是否返回准确距离
                dist_left = cv2.pointPolygonTest(hull_left, (xx, yy), measureDist=True)
                # [cy]:获得(xx,yy)到右外凸包的距离，正数说明在内部，measureDist:是否返回准确距离
                dist_right = cv2.pointPolygonTest(hull_right, (xx, yy), measureDist=True)
                # [cy]:获得(xx,yy)到内凸包的距离，正数说明在内部，measureDist:是否返回准确距离
                dist_inside = cv2.pointPolygonTest(hull_inside, (xx, yy), measureDist=True)
                # [cy]:判断在外边缘则加入边缘点集
                outside_edge_judge(xx, yy, dist_left, dist_right, outside_edge_points, 0.5, image)
                # [cy]:判断在内边缘则加入边缘点集
                inside_edge_judge(xx, yy, dist_inside, inside_edge_points, 1, image)
                # [cy]:(在外嘴唇左凸包以内或在外嘴唇右凸包以内)且在内嘴唇凸包以外为嘴唇
                if dist_left >= -0.5 or dist_right >= -0.5:
                    if dist_inside < 20:
                        lip_points.append([yy, xx])
        often_color = most_frequently_hsv(image, lip_points)  # [cy]:得到最频繁的颜色组合
        outside_edge_points = side_edge_move(outside_edge_points, 'outside', often_color, a1, a2, a3, a4, image)
        padding_inside_points(shape, inside_edge_points)  # [cy]:补齐内边缘点，以内嘴角和外嘴角之间为准
        inside_edge_points = side_edge_move(inside_edge_points, 'inside', often_color, b1, b2, b3, b4, image)
        # cv2.rectangle(image, (xstart, ystart-100), (xend, yend-100), often_color, thickness=2)  # [cy]:显示在图像上
        for pixel in lip_points:
            yy = pixel[0]
            xx = pixel[1]
            on_lip = False
            if xx in inside_edge_points.keys():
                if outside_edge_points[xx][0] < yy < inside_edge_points[xx][0] or \
                        inside_edge_points[xx][1] < yy < outside_edge_points[xx][1]:
                    on_lip = True
            else:
                if outside_edge_points[xx][0] < yy < outside_edge_points[xx][1]:
                    on_lip = True
            if on_lip:
                image[yy, xx][0] = color[0]
                image[yy, xx][1] = color[1]
                # image[yy, xx][2] += 10
        # outline(outside_edge_points, image, often_color)
        # outline(inside_edge_points, image, often_color)

    # for pixel in exact_points:
    #     yy = pixel[0]
    #     xx = pixel[1]
    #     image[yy, xx][0] = color[0]
    #     image[yy, xx][1] = color[1]
    #     image[yy, xx][2] = 255
    # for pixel in shape:
    #     xx = pixel[0]
    #     yy = pixel[1]
    #     image[yy, xx][0] = 0
    #     image[yy, xx][1] = 0
    #     image[yy, xx][2] = 255

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)  # [cy]:重新将图像转为BGR格式
    # image = cv2.GaussianBlur(image, (7, 7), 0)  # [cy]:高斯模糊图像，我认为没有用处
    # [cy]:可以查看凹多边形分成2个凸多边形
    # shape=shapes[0]
    # polygon1 = np.concatenate((shape[0:4], shape[7:12]))  # 48 49 50 51  55 56 57 58 59
    # polygon2 = shape[3:8] # 51 52 53 54 55
    # cv2.polylines(image, [polygon1], True, (0, 0, 0), 2)
    # cv2.polylines(image, [polygon2], True, (0, 0, 250), 2)
    return image


def update(h,s):
    input_image_path = "C:/Users/mibbp/Pictures/Lipstick.jpg"  # [cy]:输入图像.jpg
    lipstick_color = [h,s, 0]  # [cy]:嘴唇颜色
    image_output = feature(input_image_path, lipstick_color)  # [cy]:处理图像
#     cv2.imshow("output", image_output)  # [cy]:显示 输出图像2.jpg
    cv2.imwrite("C:/Users/mibbp/Pictures/Lipstickoutput.jpg" , image_output)  # [cy]:保存 输出图像2.jpg
    # cv2.waitKey(0)

def nothing(x):
    pass

def test():
    print('Lipstick11')

def main(h, s):

    if(h>180):
        h-=180

    update(h,s)
    cv2.waitKey(0)



if __name__=="__main__":

    main()

