import os
import cv2
import numpy as np
from pose_transform import Transform


def get_keep_aspect_ratio_info(dh, dw, sh, sw):
    h_ratio = float(dh) / sh
    w_ratio = float(dw) / sw
    if h_ratio >= 1.0 and w_ratio >= 1.0:
        ratio = 1.0
        rh = sh
        rw = sw
    elif h_ratio > w_ratio:
        ratio = w_ratio
        rw = dw
        rh = w_ratio * sh
    else:
        ratio = h_ratio
        rh = dh
        rw = h_ratio * sw
    return ratio, int(rh), int(rw)


def keep_aspect_ratio_resize(img, dh, dw):
    sh, sw, sc = img.shape
    ratio, rh, rw = get_keep_aspect_ratio_info(dh, dw, sh, sw)
    im_data = cv2.resize(img, (rw, rh))
    data = np.zeros([dh, dw, 3], dtype=np.uint8)
    data[:rh, :rw, :] = im_data
    return data, ratio


def drawlines(img1, img2, lines1, pts1, pts2):

    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        h1, w1, c1 = img1.shape
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [w1, -(r[2] + r[0] * w1) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 3)
        img1 = cv2.circle(img1, tuple(pt1), 6, color, 5)
        img2 = cv2.circle(img2, tuple(pt2), 6, color, 5)
    return img1, img2


def draw_epipolar_line(im1color, im2color, fp1, fp2, F2to1, save_file, num=10, file1=None, file2=None):
    orb = cv2.ORB_create(50)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    im1gray = cv2.cvtColor(im1color, cv2.COLOR_BGR2GRAY)
    im2gray = cv2.cvtColor(im2color, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(im1gray, None)
    kp2, des2 = orb.detectAndCompute(im2gray, None)
    if len(kp1) == 0:
        print(f"0 kps: {file1}")
        return

    if len(kp2) == 0:
        print(f"0 kps: {file2}")
        return

    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = []
    pts2 = []
    homo_pts2 = []
    for mc in matches[:num]:
        pt1 = kp1[mc.queryIdx].pt
        pt2 = kp2[mc.trainIdx].pt
        pts1.append(pt1)
        pts2.append(pt2)
        homo_pts2.append([pt2[0], pt2[1], 1])

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    homo_pts2 = np.int32(homo_pts2)

    lines1 = F2to1.dot(homo_pts2.T)
    lines1 = lines1.T

    img1, img2 = drawlines(im1color, im2color, lines1, pts1, pts2)
    h1, w1, c1 = im1color.shape
    h2, w2, c2 = im2color.shape
    h_ = max(h1, h2)
    w_ = max(w1, w2)
    img1, _ = keep_aspect_ratio_resize(img1, h_, w_)
    img2, _ = keep_aspect_ratio_resize(img2, h_, w_)
    im_v = cv2.vconcat([img1, img2])
    im_v = cv2.putText(im_v, fp1, [50, 50], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    im_v = cv2.putText(im_v, fp2, [50, 100], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_file, im_v)


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def get_f1to2(extrinsic1, extrinsic2, intrinsic1, intrinsic2, rot_thr=80.0):
    # extrinsic1, extrinsic2: world to camera
    relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
    R = relative[:3, :3]
    # remove pairs that have a relative rotation angle larger than 80 degrees
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
    if theta > rot_thr:
        return None

    T = relative[:3, 3]
    tx = skew(T)
    E_gt = np.dot(tx, R)
    F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))
    return F_gt


def cv_resize(im, scale):
    h, w, c = im.shape
    nh = int(h * scale)
    nw = int(w * scale)
    return cv2.resize(im, dsize=(nw, nh))


def visualize_ep(
    imf1: str,
    imf2: str,
    pose1: np.ndarray,
    pose2: np.ndarray,
    intrinsic1: np.ndarray,
    intrinsic2: np.ndarray,
    im1: np.ndarray = None,
    im2: np.ndarray = None,
    scale1: float = None,
    scale2: float = None,
):
    # extrinsic1, extrinsic2: camera to world
    pose1 = Transform(mat=pose1).inverse().matrix
    pose2 = Transform(mat=pose2).inverse().matrix
    F2to1 = get_f1to2(pose2, pose1, intrinsic2, intrinsic1)
    if im1 is None:
        im1 = cv2.imread(imf1)
    if im2 is None:
        im2 = cv2.imread(imf2)
    if scale1:
        im1 = cv_resize(im1, scale1)
    if scale2:
        im2 = cv_resize(im2, scale2)
    name1 = os.path.basename(imf1)
    name2 = os.path.basename(imf2)
    save_file = f"./output/{name1}_{name2}.jpg"
    draw_epipolar_line(im1color=im1, im2color=im2, fp1=name1, fp2=name2, F2to1=F2to1, save_file=save_file, num=20)
