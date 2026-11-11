"""Independent source RGB port and camera-projection evidence for cap ambiguity."""
import cv2
import numpy as np


def classify(rgb):
    a=cv2.resize(rgb,(800,800),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    background=float(np.median(np.r_[gray[200:480,:55].ravel(),gray[200:480,-55:].ravel()]))
    foreground=gray<background-20
    widths=foreground[:,80:720].sum(axis=1);candidate=(widths>400)&(np.arange(800)>240)&(np.arange(800)<480)
    rows=np.flatnonzero(candidate)
    proof=dict(face=None,method='source_port_hole_groups_or_camera_projection',background=background)
    if len(rows)<8 or len(rows)>110:return dict(proof,reason='No bounded horizontal phone body band')
    lo,hi=int(rows.min()),int(rows.max())+1;line=foreground[rows[np.argmax(widths[rows])]];xs=np.flatnonzero(line)
    if not len(xs):return dict(proof,reason='No source phone extent')
    x0,x1=int(xs.min()),int(xs.max())+1;bw=x1-x0
    if not 400<bw<720 or x0<40 or x1>760:return dict(proof,reason='Clipped or unsupported horizontal extent')
    # Boundaries from source contrast; the two outer ends are independent of
    # center paper, port and camera features being classified.
    xx=np.arange(x0,x1);first=np.argmax(foreground[max(180,lo-35):hi,xx],axis=0)+max(180,lo-35)
    distance=np.minimum(xx-x0,x1-1-xx);end=(distance>=.025*bw)&(distance<=.05*bw)
    if end.sum()<12:return dict(proof,reason='Insufficient exposed cap ends')
    rail=np.polyfit(xx[end],first[end],1);error=float(np.percentile(np.abs(first[end]-np.polyval(rail,xx[end])),95))
    if error>2.5:return dict(proof,reason='Exposed source upper rail lacks consensus')
    middle=(xx>x0+.12*bw)&(xx<x1-.12*bw);raised=np.polyval(rail,xx)-first
    halves=np.array_split(np.flatnonzero(middle),2);projection=[float(np.mean(raised[h]>3)) for h in halves]
    camera=(min(projection)>.3 and float(np.percentile(raised[middle],95))<.12*bw)
    # Require a complete port and two drilled-hole groups at more than one
    # source intensity level. Long horizontal bevels cannot supply these shapes.
    patch=gray[max(180,lo-12):hi+4,x0:x1];origin=max(180,lo-12)
    trials=[];detections=[]
    for threshold in (20,30,40,50,60):
        binary=cv2.morphologyEx((patch<threshold).astype(np.uint8)*255,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);small=[];ports=[]
        for c in contours:
            x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c);cx=x+w/2;cy=y+h/2+origin
            if x<.06*bw or x+w>.94*bw or cy<lo+1 or cy>hi-1:continue
            if 3<=w<=.03*bw and 3<=h<=.03*bw and .5<w/h<1.8 and area>=5:small.append([float(cx),float(cy),w,h])
            if .06*bw<=w<=.16*bw and 8<=h<=.5*(hi-lo) and 1.7<w/h<6 and abs(cx-bw/2)<.12*bw and area>=30:ports.append([float(cx),float(cy),w,h])
        holes=[]
        for port in ports:
            px,py,pw,ph=port;near=[p for p in small if abs(p[1]-py)<=4 and (p[0]<px-pw or p[0]>px+pw)]
            left=[p for p in near if p[0]<px-pw];right=[p for p in near if p[0]>px+pw]
            if len(left)>=3 and len(right)>=3:holes.append(dict(port=port,left=left,right=right))
        trials.append(dict(threshold=threshold,ports=ports,small_holes=small,bottom_hardware=holes))
        if len(holes)==1:detections.append(dict(threshold=threshold,**holes[0]))
    consistent=[]
    for first in detections:
        matched=[v for v in detections if np.linalg.norm(np.array(v['port'][:2])-np.array(first['port'][:2]))<=2]
        if len(matched)>len(consistent):consistent=matched
    bottom=len(consistent)>=2
    proof.update(body_bounds=[x0,lo,x1,hi],upper_rail_residual_p95_px=error,upper_rail=rail.tolist(),camera_projection_fractions=projection,camera_projection=camera,port_trials=trials,consistent_port_thresholds=[v['threshold'] for v in consistent])
    if bottom and not camera:proof['face']='bottom'
    elif camera and not bottom:proof['face']='top'
    else:proof['reason']='Independent cap evidence ambiguous or incomplete'
    return proof
