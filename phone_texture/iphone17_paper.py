"""Read the two long paper sides and the lower edge directly from the bottom photo."""
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


def _side_lines(gray, seed):
    h,w = gray.shape
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return []
    sides = []
    for x,y,u,v in lines.reshape(-1,4):
        if abs(y-v)<h*.28 or abs(x-u)>.13*abs(y-v):
            continue
        if min(y,v)>.5*h or max(y,v)<.46*h:
            continue
        slope = float((x-u)/(y-v))
        intercept = float(x-slope*y)
        middle = slope*h*.5+intercept
        if seed[0]-.34*h<middle<seed[2]+.34*h:
            sides.append((middle,slope,intercept))
    return sorted(sides)


def _paper_ends(gray,left,right):
    h,w = gray.shape
    gx = np.gradient(ndimage.gaussian_filter(gray.astype(float),.7),axis=1)
    gy = -np.gradient(ndimage.gaussian_filter(gray.astype(float),.7),axis=0)
    x0,x1 = round(left[0]),round(right[0])
    if x0<12 or x1>w-12:
        return []
    # Both long sides need a continuous photographic edge in held-out rows.
    yy = np.arange(round(.14*h),round(.69*h))
    for _,s,b in (left,right):
        xx = np.rint(s*yy+b).astype(int)
        profile = abs(gx[yy[:,None],xx[:,None]+np.arange(-2,3)]).max(axis=1)
        if np.percentile(profile,20)<2.5:
            return []
    # Printed strokes lie on paper and are present at several heights.
    inside = gray[yy,x0+5:x1-4]
    if np.mean(inside<135)<.035 or np.percentile(inside,75)<170:
        return []
    xs = np.arange(x0+5,x1-4)
    mid = (x0+x1)/2
    candidates = []
    for slope in np.linspace(-.12,.12,13):
        offsets = np.rint(slope*(xs-mid)).astype(int)
        rows = np.arange(round(.63*h),round(.94*h))
        values = gy[rows[:,None]+offsets,xs]
        response = np.median(values,axis=1)
        peaks = find_peaks(response,height=4)[0]
        for k in peaks:
            y = rows[k]
            # The paper edge ends at its corners. Reject a metal bevel of
            # comparable strength extending through the exterior on both sides.
            outside = np.r_[x0-10:x0-3,x1+4:x1+11]
            outer = gy[y+np.rint(slope*(outside-mid)).astype(int),outside]
            if response[k] < np.median(np.maximum(outer,0))*1.35+2:
                continue
            predicted = y+offsets
            window = predicted[:,None]+np.arange(-3,4)
            field = gy[window,xs[:,None]]
            ends = window[np.arange(len(xs)),field.argmax(axis=1)]
            good = field.max(axis=1)>3
            if good.mean()<.85 or np.percentile(abs(ends-predicted),80)>2:
                continue
            # Match each end of the measured lower edge to an actual side.
            finite = True
            for _,s,b in (left,right):
                yy0 = np.arange(y-12,y-4)
                yy1 = np.arange(y+5,min(h,y+12))
                def strength(ys):
                    xx = np.rint(s*ys+b).astype(int)
                    return np.median(abs(gx[ys[:,None],xx[:,None]+np.arange(-2,3)]).max(axis=1))
                if not len(yy1) or strength(yy0)<2 or strength(yy1)>strength(yy0)*.8:
                    finite = False
            if finite:
                candidates.append((float(response[k]),int(y),xs,ends))
    return candidates



def _paper_tabs(gray, mask, left, right, end_y):
    """Include a bounded paper tab extending past a long side onto darker metal."""
    h,w = gray.shape
    rows = np.arange(h)
    for (_,s,b),direction in ((left,-1),(right,1)):
        start = np.rint(s*rows+b).astype(int)
        # The outside metal and neighbouring paper are sampled independently
        # on every row to avoid including a continuous reflection or bevel.
        def strip(offsets):
            xs = start[:,None]+direction*np.asarray(offsets)
            return gray[rows[:,None],np.clip(xs,0,w-1)].astype(float)
        metal = np.median(strip(np.arange(10,15)),axis=1)
        white = np.percentile(strip(np.arange(-13,-2)),80,axis=1)
        values = strip(np.arange(10))
        supported = ((values-metal[:,None])>.58*(white-metal)[:,None]) & ((white-metal)[:,None]>35)
        supported &= rows[:,None] < end_y-2
        labels,count = ndimage.label(ndimage.binary_closing(supported,structure=np.ones((3,2))))
        for label in range(1,count+1):
            ys,depth = np.nonzero(labels==label)
            if not len(ys) or depth.max()<3 or depth.min()>1 or depth.max()>=8:
                continue
            # A small protrusion must start and end within the measured side.
            extended = np.unique(ys[depth>=3])
            if not 3<=len(extended)<.25*h or np.ptp(extended)>len(extended)+2:
                continue
            for y in extended:
                extent = depth[ys==y].max()
                x1,x2 = sorted((start[y],start[y]+direction*(extent+1)))
                mask[y,max(0,x1):min(w,x2+1)]=True
    return mask

def measured_paper(a, proof):
    gray = cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    h,w = gray.shape
    sides = _side_lines(gray,proof['seed_box'])
    candidates = []
    for j,left in enumerate(sides):
        for right in sides[j+1:]:
            if not .44*h<right[0]-left[0]<.73*h or abs(right[1]-left[1])>.08:
                continue
            for evidence in _paper_ends(gray,left,right):
                candidates.append((*evidence,left,right))
    if not candidates:
        raise ValueError('Paper lacks independently supported side and lower edges')
    candidates.sort(key=lambda v:-v[0])
    score,y,xs,ends,left,right = candidates[0]
    if any(v[0]>.75*score and (abs(v[1]-y)>4 or abs(v[4][0]-left[0])>4 or
           abs(v[5][0]-right[0])>4) for v in candidates[1:]):
        raise ValueError('Competing photographed paper boundaries')
    xx = np.arange(w)
    yy = np.arange(h)[:,None]
    lo = (left[1]*yy+left[2])
    hi = (right[1]*yy+right[2])
    end = np.interp(xx,xs,ndimage.median_filter(ends,size=3))
    mask = (xx>=lo-1)&(xx<=hi+1)&(yy<=end+1)&(a[...,3]>0)
    mask = _paper_tabs(gray,mask,left,right,y) & (a[...,3]>0)
    return mask, dict(proof, method='source_print_and_three_photographed_edges',
                      end_method='native_lower_edge_with_exterior_metal_holdouts',
                      source_strip_edges=[list(left[1:]),list(right[1:])],
                      source_end_columns=xs.tolist(),source_end_y_by_column=(end[xs]+1).tolist(),
                      end_response=score,pixels=int(mask.sum()))
