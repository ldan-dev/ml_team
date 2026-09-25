import io
import zipfile
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = 'MPEG7dataset.zip'
CLASSES = ('Bone','bottle','brick','butterfly','fork','Heart','guitar','octopus','tree','watch')
SEED = 20260925
N = 5

def normalize(im):
    raw=np.asarray(im.convert('L'))
    mask=(raw>127).astype(np.uint8)*255
    ys,xs=np.where(mask>0)
    y0,y1=ys.min(),ys.max()+1
    x0,x1=xs.min(),xs.max()+1
    crop=mask[y0:y1,x0:x1]
    scale=64/max(crop.shape)
    w,h=max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))
    small=cv2.resize(crop,(w,h),interpolation=cv2.INTER_NEAREST)
    out=np.zeros((128,128),np.uint8)
    out[(128-h)//2:(128-h)//2+h,(128-w)//2:(128-w)//2+w]=small
    return out

def transform(base,rng):
    angle=float(rng.uniform(0,360))
    scale=float(rng.uniform(.65,1.1))
    dx,dy=rng.uniform(-15,15,2)
    M=cv2.getRotationMatrix2D((64,64),angle,scale)
    M[:,2]+=(dx,dy)
    out=cv2.warpAffine(base,M,(128,128),flags=cv2.INTER_NEAREST)
    out=cv2.resize(out,(64,64),interpolation=cv2.INTER_AREA)
    return (out>=128).astype(np.uint8), (angle,scale,dx,dy)

def hu(im):
    h=cv2.HuMoments(cv2.moments(im,binaryImage=True)).ravel()
    return -np.sign(h)*np.log10(np.maximum(np.abs(h),1e-30))

def main():
    source=zipfile.ZipFile(SOURCE)
    rng=np.random.default_rng(SEED)
    entries=[]
    names=source.namelist()
    for cl in CLASSES:
        files=sorted(n for n in names if n.rsplit('/',1)[-1].startswith(cl+'-') and n.lower().endswith('.gif'))
        assert len(files)==20,(cl,len(files))
        for name in files:
            im=Image.open(io.BytesIO(source.read(name)))
            base=normalize(im)
            for rep in range(N):
                mask,pose=transform(base,rng)
                entries.append((cl,name,rep,mask,pose))
    original=np.array(sorted(set(e[1] for e in entries)))
    origin_labels=[next(e[0] for e in entries if e[1]==file) for file in original]
    tr,te=train_test_split(original,test_size=.3,stratify=origin_labels,random_state=42)
    tr=set(tr);te=set(te)
    X=np.array([e[3].ravel() for e in entries]); H=np.array([hu(e[3]) for e in entries]);
    y=np.array([e[0] for e in entries]);
    a=np.array([i for i,e in enumerate(entries) if e[1] in tr]);b=np.array([i for i,e in enumerate(entries) if e[1] in te])
    print('n',len(y),'train',len(a),'test',len(b))
    for title,data in [('pixels',X),('Hu',H)]:
        for clf in [KNeighborsClassifier(),SVC()]:
            model=make_pipeline(StandardScaler(),clf) if title=='Hu' else clf
            model.fit(data[a],y[a]);pred=model.predict(data[b]);
            print(title,clf.__class__.__name__,round(accuracy_score(y[b],pred),4),int(sum(pred==y[b])))
    return entries,tr,te

if __name__=='__main__': main()
