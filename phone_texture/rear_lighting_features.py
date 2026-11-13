"""Source-only evidence for a dark rear emblem, never a pixel protection mask."""
import cv2
import numpy as np
from scipy import ndimage


def dark_rear_mark(a, panel):
    """Require a stable central body and a separate leaf over three thresholds.

    This scoped detector handles high-contrast dark marks in the 17-series
    capture profile. Low-contrast/ambiguous marks are left to the existing
    path, with the reason recorded. No corrected RGB or stored SKU box enters.
    """
    h,w=a.shape[:2];yy,xx=np.indices((h,w))
    search=panel&(a[...,3]>=240)&(xx>w*.28)&(xx<w*.72)&(yy>h*.48)&(yy<h*.76)
    empty=np.zeros((h,w),bool)
    proof=dict(method='source_dark_emblem_threshold_consensus_v1',detected=False,
               usage='illumination_fit_exclusion_only',pixel_protection=False)
    if search.sum()<100:return empty,dict(proof,reason='insufficient_panel')
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(float)
    background=float(np.percentile(gray[search],70));readings=[]
    for deficit in (15,25,35):
        dark=(gray<background-deficit)&search
        n,labels,stats,centres=cv2.connectedComponentsWithStats(dark.astype('uint8'),8)
        bodies=[]
        for i in range(1,n):
            x,y,bw,bh,area=map(int,stats[i]);cx,cy=centres[i]
            if (.12*w<bw<.28*w and .065*h<bh<.13*h and
                    .45<area/(bw*bh)<.9 and .38*w<cx<.62*w):
                bodies.append(i)
        if len(bodies)!=1:return empty,dict(proof,reason='no_unique_high_contrast_body')
        i=bodies[0];x,y,bw,bh,_=map(int,stats[i]);leaves=[]
        for k in range(1,n):
            lx,ly,lw,lh,area=map(int,stats[k]);cx,cy=centres[k]
            if (k!=i and x+bw*.25<cx<x+bw*.85 and y-bh*.55<cy<y and
                    .10*bw<lw<.4*bw and .10*bh<lh<.4*bh and area>20):
                leaves.append(k)
        if len(leaves)!=1:return empty,dict(proof,reason='no_unique_leaf')
        readings.append((labels==i)|(labels==leaves[0]))
    intersection=np.logical_and.reduce(readings);union=np.logical_or.reduce(readings)
    agreement=float(intersection.sum()/max(1,union.sum()))
    if agreement<.94:return empty,dict(proof,reason='unstable_material_boundary',agreement=agreement)
    # Include the observed sensor transition, without freezing the emblem.
    margin=max(2,int(round(w*.002)))
    mark=ndimage.binary_dilation(union,iterations=margin)&panel
    ys,xs=np.nonzero(mark)
    return mark,dict(proof,detected=True,agreement=agreement,pixels=int(mark.sum()),
                     bounds=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                     source_background_gray=background,threshold_deficits=[15,25,35])
