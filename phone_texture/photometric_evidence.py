import numpy as np
from scipy.special import erf

def evidence(a,pair,col,degree):
 y0,y1=pair;width=y1-y0;ys=np.arange(max(0,int(y0-width)),min(len(a),int(y1+width)+1));z=(ys-(y0+y1)/2)/width
 steps=[.5*(1+erf((ys-y)/(2**.5))) for y in [y0,y1]];X=np.column_stack([*[z**i for i in range(degree+1)],*steps]);Y=a[ys,col,:3].astype(float);beta=np.linalg.lstsq(X,Y,rcond=None)[0];err=Y-X@beta;var=max(1/12,float((err**2).sum())/max(1,3*(len(ys)-X.shape[1])));cov=np.linalg.inv(X.T@X);vectors=beta[-2:];norms=np.linalg.norm(vectors,axis=1);ts=norms/(np.sqrt(3*var*np.diag(cov)[-2:]));agreement=float(-vectors[0]@vectors[1]/(norms.prod()+1e-12))
 rectangular=steps[0]-steps[1]
 R=np.column_stack([*[z**i for i in range(degree+1)],rectangular]);rb=np.linalg.lstsq(R,Y,rcond=None)[0]
 re=Y-R@rb;rv=max(1/12,float((re**2).sum())/max(1,3*(len(ys)-R.shape[1])));rse=np.sqrt(rv*np.linalg.inv(R.T@R)[-1,-1])
 joint=float(np.linalg.norm(rb[-1])/(np.sqrt(3)*rse))
 return dict(edge_t=ts.tolist(),joint_rectangular_t=joint,opposing_agreement=agreement,residual_rms=float(np.sqrt(var)),source_steps=vectors.tolist())
