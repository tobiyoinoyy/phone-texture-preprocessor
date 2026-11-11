import numpy as np
from PIL import Image,ImageDraw
from phone_texture.bottom_rectify import PROFILES,project

def fixture(model,mirrored=False,shift=0,missing=False,wrong_row=False,distort=False,no_screw=False,large=False,near_center=False,inner_shift=None,outer_gray=180,label_edge=False,missing_edge_rows=False,label_occlusion=False):
    p=PROFILES[model];w,h=1030,140
    y,x=np.mgrid[:h,:w]
    texture=(210+9*np.sin(x*.17)+6*np.cos(y*.31)).astype('uint8')
    a=np.dstack([texture,texture,texture,np.full((h,w),255,dtype='uint8')])
    im=Image.fromarray(a);d=ImageDraw.Draw(im)
    params=[1030,514.5 if near_center else (570 if large else 530),.025]
    holes=np.array(p['holes']);screws=np.array(p['screws']);center=p['center']
    if mirrored:holes=np.sort(p['width']-holes);screws=np.sort(p['width']-screws);center=p['width']-center
    def loc(mm):return float(project(params,mm/p['width']-.5))
    for i,mm in enumerate(holes):
        if missing and i==1:continue
        xx=loc(mm)+(4*np.sin(i*1.7) if distort else 0);yy=70+(5 if wrong_row and i==2 else 0)
        d.ellipse((round(xx-10),yy-10,round(xx+10),yy+10),fill='#252525')
    xx=loc(center)+shift
    d.rounded_rectangle((round(xx-65),48,round(xx+65),92),radius=18,
                        fill=(outer_gray,)*3 if inner_shift is not None else '#202020')
    if inner_shift is not None:
        ix=loc(center)+inner_shift
        d.rounded_rectangle((round(ix-52),54,round(ix+52),86),radius=12,fill='#202020')
    if not no_screw:
        xx=loc(screws[1]);d.ellipse((round(xx-11),59,round(xx+11),81),fill='#353535')
    xx=loc(center)
    if label_edge or label_occlusion:d.rectangle((round(xx-90),0,round(xx-(55 if label_occlusion else 75)),h-1),fill='#ffffff')
    if missing_edge_rows:d.rectangle((round(xx-72),65,round(xx-45),75),fill='#d2d2d2')
    return np.asarray(im),center/p['width']*(w-1)
