"""Physical-domain Jacobian checks, using the geometric ANTs implementation."""
import numpy as np

def checked_logjac(determinant,mask,bounds=(-2.,2.)):
 d=np.asarray(determinant);m=np.asarray(mask,bool)
 if d.shape!=m.shape or not m.any():raise ValueError('JACOBIAN_MASK_GEOMETRY')
 if not np.isfinite(d[m]).all() or np.any(d[m]<=0):raise ValueError('NONPOSITIVE_OR_NONFINITE_BRAIN_JACOBIAN')
 log=np.log(d[m]);out=np.zeros(d.shape,np.float32);out[m]=np.clip(log,*bounds)
 return out,{'logjac_min':float(log.min()),'logjac_max':float(log.max()),'logjac_mean':float(log.mean()),'logjac_sd':float(log.std()),'clipped_fraction':float(np.mean((log<bounds[0])|(log>bounds[1])))}

def geometry_selftest(folder):
 import ants
 from pathlib import Path
 p=Path(folder);p.mkdir(parents=True,exist_ok=True);shape=(24,24,24)
 # Test anisotropic physical spacing and oblique axes, not just unit-grid coordinates.
 angle=.31;direction=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
 spacing=(1.5,2.,2.5);origin=(-18.,-20.,-25.)
 indices=np.moveaxis(np.indices(shape),0,-1)
 xyz=np.einsum('ij,...j->...i',direction,indices*np.array(spacing))+np.array(origin)
 fixed=ants.from_numpy(np.zeros(shape,np.float32),origin=origin,spacing=spacing,direction=direction)
 rows=[]
 for name,scale,translation in [('identity',1.,(0,0,0)),('translation',1.,(2.,-4.,3.)),('expansion',1.1,(0,0,0)),('contraction',.9,(0,0,0))]:
  field=((scale-1)*xyz+np.array(translation)).astype(np.float32)
  warp=ants.from_numpy(field,origin=origin,spacing=spacing,direction=direction,has_components=True)
  path=p/(name+'.nii.gz');ants.image_write(warp,str(path))
  jac=ants.create_jacobian_determinant_image(fixed,str(path),do_log=False,geom=True).numpy()[3:-3,3:-3,3:-3]
  error=float(np.max(np.abs(jac-scale**3)))
  if error>2e-4:raise ValueError('SYNTHETIC_PHYSICAL_JACOBIAN_FAILED:'+name+':'+str(error))
  rows.append({'case':name,'expected_determinant':scale**3,'max_absolute_error':error})
 return rows
