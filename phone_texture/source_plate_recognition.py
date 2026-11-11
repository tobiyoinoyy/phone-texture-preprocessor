"""Disjoint native-RGB identity witness after frozen outline registration."""
import itertools
import cv2
import numpy as np


def camera_triad(rgb,matrix):
    scale=8;size=48*scale
    a=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)
    transform=np.array(matrix,dtype=np.float32).copy();transform[:,2]*=scale
    a=cv2.warpAffine(a,transform,(size,size),flags=cv2.INTER_LINEAR)
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    x0,x1,y0,y1=16*scale,33*scale,7*scale,20*scale
    patch=gray[y0:y1,x0:x1];detections=[];trials=[]
    body_width=18*scale
    for threshold in (20,30,40,50,60,80):
        mask=(patch<threshold).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);circles=[]
        for contour in contours:
            area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
            x,y,w,h=cv2.boundingRect(contour);(cx,cy),radius=cv2.minEnclosingCircle(contour)
            if x==0 or y==0 or x+w>=patch.shape[1] or y+h>=patch.shape[0]:continue
            if not .055*body_width<radius<.14*body_width or not .7<w/max(h,1)<1.4:continue
            circularity=4*np.pi*area/max(perimeter*perimeter,1)
            if circularity<.65 or area/(np.pi*radius*radius)<.65:continue
            circles.append(dict(x=float(cx+x0),y=float(cy+y0),r=float(radius),circularity=float(circularity)))
        valid=[]
        for group in itertools.combinations(circles,3):
            left=sorted(group,key=lambda c:c['x']);pair=sorted(left[:2],key=lambda c:c['y']);right=left[2];r=float(np.mean([c['r'] for c in group]));upper,lower=pair
            if max(c['r'] for c in group)/min(c['r'] for c in group)>1.4:continue
            if abs(upper['x']-lower['x'])>.6*r:continue
            if not 1.5*r<lower['y']-upper['y']<4*r:continue
            if not 1.4*r<right['x']-(upper['x']+lower['x'])/2<3.6*r:continue
            if abs(right['y']-(upper['y']+lower['y'])/2)>.6*r:continue
            valid.append([upper,lower,right])
        trials.append(dict(threshold=threshold,circles=circles,triads=len(valid)))
        if len(valid)==1:detections.append(dict(threshold=threshold,centers=[[c['x'],c['y']] for c in valid[0]],radii=[c['r'] for c in valid[0]]))
    consistent=[]
    for first in detections:
        matches=[x for x in detections if np.max(np.linalg.norm(np.array(first['centers'])-np.array(x['centers']),axis=1))<=2]
        if len(matches)>len(consistent):consistent=matches
    return dict(accepted=len(consistent)>=2,method='three_round_source_lenses_in_ordered_triangle_at_multiple_thresholds',
        independent_of_atlas_class_scores=True,consistent_thresholds=[x['threshold'] for x in consistent],
        detections=detections,trials=trials)
