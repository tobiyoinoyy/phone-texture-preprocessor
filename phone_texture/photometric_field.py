"""Cross-validated source-only photometric field; no output pixels changed."""
import numpy as np
class AnalyticalRGBA:
 def __init__(self,data):self.data=np.asarray(data,float);self.height,self.width=self.data.shape[:2];self.size=(self.width,self.height)
 def convert(self,mode):
  if mode!='RGBA':raise ValueError('Analytical evidence is RGBA only')
  return self
 def __array__(self,dtype=None,copy=None):return np.array(self.data,dtype=dtype,copy=True) if copy else np.asarray(self.data,dtype=dtype)

def normalize(image,rows):
 a=np.asarray(image.convert('RGBA')).astype(float);h,w=a.shape[:2];result=a.copy();proof=[]
 for band in (0,1):
  lo,hi=rows[2*band:2*band+2];width=hi-lo;lower=max(0,int(lo-3*width));upper=min(h,int(hi+3*width)+1);evaluation=np.arange(lower,upper);ez=(evaluation-(lo+hi)/2)/width;models=[]
  for degree in (1,2):
   for span in (.5,1.,1.5,2.):
    ys=np.arange(max(0,int(lo-width*span)),min(h,int(hi+width*span)+1));ys=ys[(ys<lo-3)|(ys>hi+3)];valid=a[ys,:,3]>=128;groups={}
    for col in range(w):groups.setdefault(valid[:,col].tobytes(),[]).append(col)
    score=np.full(w,np.inf);se=np.full(w,np.inf);pred=np.zeros((len(evaluation),w,3));support=np.zeros((w,2),int)
    for cols in groups.values():
     selected=ys[valid[:,cols[0]]];counts=[int(np.sum(selected<lo)),int(np.sum(selected>hi))]
     if min(counts)<4:continue
     z=(selected-(lo+hi)/2)/width;X=np.column_stack([z**i for i in range(degree+1)]);Y=a[selected[:,None],np.array(cols)[None,:],:3].reshape(len(selected),-1);common=(selected>=lo-.5*width)&(selected<=hi+.5*width);losses=[]
     for fold in (0,1):
      held=common&((selected//2)%2==fold);train=~held
      if held.sum()<2 or train.sum()<degree+3:continue
      coef=np.linalg.lstsq(X[train],Y[train],rcond=None)[0];loss=((Y[held]-X[held]@coef)**2).reshape(int(held.sum()),len(cols),3).mean(axis=2);losses.append(loss)
     if not losses:continue
     loss=np.concatenate(losses);score[cols]=loss.mean(axis=0);se[cols]=loss.std(axis=0,ddof=1)/np.sqrt(len(loss));coef=np.linalg.lstsq(X,Y,rcond=None)[0];design=np.column_stack([ez**i for i in range(degree+1)]);pred[:,cols]=(design@coef).reshape(len(evaluation),len(cols),3);support[cols]=counts
    models.append(dict(degree=degree,span=span,score=score,se=se,pred=pred,support=support))
  choices=[]
  for col in range(w):
   eligible=[m for m in models if np.isfinite(m['score'][col])]
   if not eligible:choices.append(None);continue
   best=min(eligible,key=lambda m:m['score'][col]);cut=best['score'][col]+best['se'][col];allowed=[m for m in eligible if m['score'][col]<=cut];chosen=min(allowed,key=lambda m:(m['degree'],-m['span'],m['score'][col]));result[lower:upper,col,:3]=a[lower:upper,col,:3]-chosen['pred'][:,col]+128
   choices.append(dict(degree=chosen['degree'],span=chosen['span'],cv_mse=float(chosen['score'][col]),cv_standard_error=float(chosen['se'][col]),support=chosen['support'][col].tolist()))
  proof.append(dict(band=band,models=choices,method='blocked_outer_source_cv_one_standard_error',background_alpha_minimum=128))
 return AnalyticalRGBA(result),proof
