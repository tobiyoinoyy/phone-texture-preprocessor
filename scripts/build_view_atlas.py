"""Build a versioned classifier atlas from explicitly labelled raw references.

Not run on import. SKU_01–06 supply reference descriptors; SKU_07–10 remain
held out. This atlas never participates in geometry, lighting or alignment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from phone_texture.view_recognition import DATA,FACES,PROFILE,read_rgb,edge_feature,side_feature


def main():
    p=argparse.ArgumentParser();p.add_argument('reference_root',type=Path);a=p.parse_args()
    arrays={};records=[]
    for n in range(1,7):
        group=f'SKU_{n:02d}'
        for face in FACES:
            path=a.reference_root/group/(face+'.JPG');rgb=read_rgb(path)
            id=group+'_'+face
            arrays[id+'_edge']=edge_feature(rgb)
            if face in ('left','right'):arrays[id+'_side']=side_feature(rgb)
            records.append(dict(id=id,group=group,face=face,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA/'atlas.npz',**arrays)
    (DATA/'atlas.json').write_text(json.dumps(dict(profile=PROFILE,references=records,
        scope='Known iPhone 17 studio raw capture; not a universal classifier',
        held_out_groups=['SKU_07','SKU_08','SKU_09','SKU_10'],
        algorithm='image-only coarse edges plus fine side hardware edges and mirror comparison',
        thresholds=dict(min_similarity=.9,min_margin=.025,min_side_similarity=.9,min_side_margin=.035,min_mirror_advantage=.015)),indent=2))
    print(DATA)


if __name__=='__main__':main()
