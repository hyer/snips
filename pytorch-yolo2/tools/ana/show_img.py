import cv2
import numpy as np
def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

img = cv2.imread("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/predictions.jpg")

neti_w = 800
neti_h = neti_w
step = 32
for i in range(25):
    for j in range(25):
        cv2.line(img, (j*step, 0), (j*step, neti_w), (50, 50, 50), 1)
        cv2.line(img, (0, j*step), (neti_w, j*step), (50, 50, 50), 1)
        # drawline(img, (j*step, 0), (j*step, neti_w), (0, 200, 0), 1)

cv2.imshow("img", img)
cv2.waitKey()