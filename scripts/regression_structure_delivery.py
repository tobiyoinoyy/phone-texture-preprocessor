"""Run the structure gate on a real six-face sample and its unaligned control.

Requires reviewed source-bound annotations; never approves visual review.
Business images and calibration paths stay outside the repository.
"""
import argparse
from pathlib import Path
from phone_texture import structure_review as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--process',type=Path,required=True)
    args=parser.parse_args()
    report=s.prepare(args.input,args.spec,args.process)
    files=s.batch_files(args.input)
    try:
        s.verify(files,files,s.read(args.spec))
    except ValueError as exc:
        negative=str(exc)
    else:
        raise ValueError('Negative control already passes; choose the known misaligned fixture')
    s.write(args.process/'regression.json',dict(unaligned_control_rejected=negative,
        candidate_pixel_checks=report['checks'],visual_review='pending',model_uv_verified=False))
    print('Misaligned control rejected; candidate numeric checks passed. Visual review still required.')


if __name__=='__main__':
    main()
