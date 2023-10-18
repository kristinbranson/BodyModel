import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors
from matplotlib.patches import Rectangle
import pickle
import h5py
import scipy
import scipy.ndimage as ndimage
import copy
#import quaternion

anglenames = ['yaw','roll','pitch']
wingangle_rename = {'roll': 'deviation', 'yaw': 'stroke', 'pitch': 'rotation'}
angleidx = {wingangle_rename[k]: i for i,k in enumerate(anglenames)}
WINGISTRANSFORMED = False

def loadmat(matfile):
  """
  Load mat file using scipy.io.loadmat or h5py.File.
  Input: 
    matfile: path to mat file
  Output:
    f: loaded mat file object
    datatype: 'scipy' or 'h5py'
  """
  assert os.path.exists(matfile)
  try:
    f = scipy.io.loadmat(matfile, struct_as_record=False)
    datatype = 'scipy'
  except NotImplementedError:
    f = h5py.File(matfile, 'r')
    datatype = 'h5py'    
  except:
    ValueError(f'could not read mat file {matfile}')
  return f, datatype

def quat2rpy(q):
  widx = 0
  xidx = 1
  yidx = 2
  zidx = 3
  roll  = np.arctan2(2.0 * (q[...,zidx] * q[...,yidx] + q[...,widx] * q[...,xidx]) , 1.0 - 2.0 * (q[...,xidx] * q[...,xidx] + q[...,yidx] * q[...,yidx]))
  pitch = np.arcsin(np.clip(2.0 * (q[...,yidx] * q[...,widx] - q[...,zidx] * q[...,xidx]),-1.,1.))
  yaw   = np.arctan2(2.0 * (q[...,zidx] * q[...,widx] + q[...,xidx] * q[...,yidx]) , - 1.0 + 2.0 * (q[...,widx] * q[...,widx] + q[...,xidx] * q[...,xidx]))
  return {'roll': roll, 'pitch': pitch, 'yaw': yaw}

def rpy2quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray):
    """Convert rpy to quaternion."""
    if type(roll) is not np.ndarray:
      roll = np.array([roll])
      pitch = np.array([pitch])
      yaw = np.array([yaw])
    sz = roll.shape
    q_roll = np.c_[np.cos(roll/2), np.sin(roll/2), np.zeros(sz), np.zeros(sz)]
    q_yaw = np.c_[np.cos(yaw/2), np.zeros(sz), np.zeros(sz), np.sin(yaw/2)]
    q_pitch = np.c_[np.cos(pitch/2), np.zeros(sz), np.sin(pitch/2), np.zeros(sz)]
    return quatmultiply(quatmultiply(q_yaw, q_pitch), q_roll)

def quatconj(q):
  return np.concatenate((q[...,[0,]],-q[...,[1,]],-q[...,[2,]],-q[...,[3,]]),axis=-1)

def quatmultiply(q,r):
  n0 = r[...,0]*q[...,0] - r[...,1]*q[...,1] - r[...,2]*q[...,2] - r[...,3]*q[...,3]
  n1 = r[...,0]*q[...,1] + r[...,1]*q[...,0] - r[...,2]*q[...,3] + r[...,3]*q[...,2]
  n2 = r[...,0]*q[...,2] + r[...,1]*q[...,3] + r[...,2]*q[...,0] - r[...,3]*q[...,1]
  n3 = r[...,0]*q[...,3] - r[...,1]*q[...,2] + r[...,2]*q[...,1] + r[...,3]*q[...,0]
  n = np.concatenate((n0[...,None],n1[...,None],n2[...,None],n3[...,None]),axis=-1)
  z = np.linalg.norm(n,axis=-1)
  n = n / z[...,None]
  return n

def rotate_vec_with_quat(vec, quat):
    """Uses unit quaternion `quat` to rotate vector `vec` according to:

        vec' = quat vec quat^-1.

    Any number of leading batch dimensions is supported.

    Technically, `quat` should be a unit quaternion, but in this particular
    multiplication (quat vec quat^-1) it doesn't matter because an arbitrary
    constant cancels out in the product.

    Broadcasting works in both directions. That is, for example:
    (i) vec and quat can be [1, 1, 3] and [2, 7, 4], respectively.
    (ii) vec and quat can be [2, 7, 3] and [1, 1, 4], respectively.

    Args:
        vec: Cartesian position vector to rotate, shape (B, 3). Does not have
            to be a unit vector.
        quat: Rotation unit quaternion, (B, 4).

    Returns:
        Rotated vec, (B, 3,).
    """
    if vec[..., :-1].size > quat[..., :-1].size:
        # Broadcast quat to vec.
        quat = np.tile(quat, vec.shape[:-1] + (1,))
    vec_aug = np.zeros_like(quat)
    vec_aug[..., 1:] = vec
    vec = quatmultiply(quat, quatmultiply(vec_aug, quatconj(quat)))
    return vec[..., 1:]

def quatseq2angvel(q,dt,theta=None):
  """
  av = quatseq2angvel(q,dt, [theta])
  q: (n,4) quaternion sequence
  dt: time step
  theta: angle to rotate the body coordinate system around the y-axis
  computes the angular velocity in the passive aka body frame
  """
  
  qdiff_passive = quatmultiply(quatconj(q[:-1]),q[1:])
  av = 2*qdiff_passive[:,1:]/dt
  
  if theta is not None:  
    qrot = np.array([np.cos(theta/2), 0, np.sin(theta/2), 0])
    av = rotate_vec_with_quat(av, qrot)
  
  return av

def quatseq2rpy(q):
  dq = quatmultiply(quatconj(q[:-1]),q[1:])
  drpy = quat2rpy(dq)
  sz = drpy['roll'].shape
  z = np.zeros((1,)+sz[1:])
  rpy = {}
  for k,v in drpy.items():
    rpy[k] = np.cumsum(np.r_[z,v],axis=0)
  return rpy

def mod2pi(x):
  return np.mod(x+np.pi,2*np.pi)-np.pi

def unwrap_angleseq(x,t0=0):
  assert (x.ndim==1)
  
  if t0 != 0:
    xpre = unwrap_angleseq(x[t0::-1])
    xpost = unwrap_angleseq(x[t0:])
    x = np.r_[xpre[::-1],xpost[1:]]
    return x

  # unwrap each non-nan sequence
  isnanx = np.r_[True,np.isnan(x),True]
  seqstarts = np.nonzero(isnanx[:-1] & (isnanx[1:] == False))[0]
  seqends = np.nonzero(isnanx[1:] & (isnanx[:-1] == False))[0]

  xout = np.zeros_like(x)
  xout[:] = np.nan
  for i in range(len(seqstarts)):
    xcurr = x[seqstarts[i]:seqends[i]]
    dx = mod2pi(xcurr[1:]-xcurr[:-1])
    xcurr = np.cumsum(np.r_[xcurr[0],dx])
    xout[seqstarts[i]:seqends[i]] = xcurr
  return xout
  
# def test_quatseq2rpy(qseq):
#   # qseq = modeldata['qpos'][traji][:,3:]
#   qseq = quatmultiply(quatconj(qseq[[0,]]),qseq)
#   rpy_seq = quatseq2rpy(qseq)
#   rpy_direct = quat2rpy(qseq)
#   fig,ax = plt.subplots(3,1,sharex=True)
#   for i,key in enumerate(rpy_seq.keys()):
#     ax[i].plot(rpy_seq[key],label='seq')
#     ax[i].plot(rpy_direct[key],label='direct')
#     ax[i].set_ylabel(key)
#   ax[0].legend()
#   qrecon_seq = rpy2quat(rpy_seq['roll'],rpy_seq['pitch'],rpy_seq['yaw'])
#   qrecon_direct = rpy2quat(rpy_direct['roll'],rpy_direct['pitch'],rpy_direct['yaw'])
#   err_seq = np.linalg.norm(qrecon_seq-qseq,axis=1)
#   err_direct = np.linalg.norm(qrecon_direct-qseq,axis=1)
#   fig = plt.figure()
#   plt.plot(err_seq,label='seq')
#   plt.plot(err_direct,label='direct')
#   plt.legend()
  
#   err = []
#   t0 = -1
#   for t in range(1000):
#     rpy0 = {k: v[[t0,]] for k,v in rpy_direct.items()}
#     rpy1 = {k: v[[t,]] for k,v in rpy_direct.items()}
#     qrecon0 = rpy2quat(rpy0['roll'],rpy0['pitch'],rpy0['yaw'])
#     qrecon1 = rpy2quat(rpy1['roll'],rpy1['pitch'],rpy1['yaw'])
#     dqrecon = quatmultiply(quatconj(qrecon0),qrecon1)
#     drpy_recon = quat2rpy(dqrecon)
#     drpy_direct = {k: rpy1[k]-rpy0[k] for k in rpy0.keys()}
#     err.append(np.linalg.norm(np.array(list(drpy_recon.values()))-np.array(list(drpy_direct.values()))))
#   plt.figure()
#   plt.plot(np.array(err))
#   # print(f'direct: {drpy_direct}')
#   # print(f'recon: {drpy_recon}')

#   rpy0init = {'roll': np.pi/6, 'pitch': np.pi/2+1e-16, 'yaw': -np.pi/8}
#   q0 = rpy2quat(rpy0init['roll'],rpy0init['pitch'],rpy0init['yaw'])
#   #q0 = qseq[1000]
#   rpy0 = quat2rpy(q0)
#   rpyaddpi = {'roll': mod2pi(rpy0['roll']+np.pi), 'pitch': mod2pi(np.pi-rpy0['pitch']), 'yaw': mod2pi(rpy0['yaw']+np.pi)}                          
#   q0recon = rpy2quat(rpy0['roll'],rpy0['pitch'],rpy0['yaw'])
#   q0addpi = rpy2quat(rpyaddpi['roll'],rpyaddpi['pitch'],rpyaddpi['yaw'])
#   errrecon = np.linalg.norm(q0recon-q0)
#   erraddpi = np.linalg.norm(q0addpi-q0)
#   nsample = 361
#   samples = np.linspace(-np.pi,np.pi,nsample)
#   allrpy = np.meshgrid(samples,samples,samples,indexing='ij')
#   q1 = rpy2quat(allrpy[0].flatten(),allrpy[1].flatten(),allrpy[2].flatten())
#   err = np.linalg.norm(q1-q0,axis=1)
#   err = err.reshape(allrpy[0].shape)
#   order = np.argsort(err.flatten())
#   bestidx = np.unravel_index(order[:100],err.shape)
#   bestrpys = [samples[x] for x in bestidx]
#   fig,ax = plt.subplots(4,1,sharex=True)
#   keys = list(rpy0.keys())
#   for i in range(3):
#     ax[i].cla()
#     ax[i].plot(-1,rpy0[keys[i]],'s')
#     ax[i].plot(bestrpys[i],'.')
#   ax[-1].cla()
#   ax[-1].plot(-1,errrecon,'s')
#   ax[-1].plot(err[*bestidx],'.-')
#   ax[-1].set_ylabel('error')
#   fig.tight_layout()
#   allrpy = None
#   q1 = None
#   err = None
#   order = None

#   nsample = 100
#   samples = np.linspace(-1,1,nsample)
#   allqs = np.meshgrid(samples,samples,samples,samples,indexing='ij')
#   allqs = np.stack([x.flatten() for x in allqs],axis=-1)
#   allqs = allqs/np.linalg.norm(allqs,axis=-1,keepdims=True)
#   rpys = quat2rpy(allqs)
#   rpyaddpi = {'roll': mod2pi(rpys['roll']+np.pi), 'pitch': mod2pi(np.pi-rpys['pitch']), 'yaw': mod2pi(rpys['yaw']+np.pi)}
#   allqsrecon = rpy2quat(rpys['roll'],rpys['pitch'],rpys['yaw'])
#   allerrrecon = np.minimum(np.linalg.norm(allqsrecon-allqs,axis=-1),np.linalg.norm(-allqsrecon-allqs,axis=-1))
#   q0addpi = rpy2quat(rpyaddpi['roll'],rpyaddpi['pitch'],rpyaddpi['yaw'])
#   allerraddpi = np.minimum(np.linalg.norm(q0addpi-allqs,axis=-1),np.linalg.norm(-q0addpi-allqs,axis=-1))
  
#   qseq = modeldata['qpos'][traji][:,3:]
#   qqseq = quaternion.as_quat_array(qseq)
#   dts = np.arange(len(qqseq))*dt
#   omega = quaternion.angular_velocity(qqseq,dts)
#   dq = quatmultiply(quatconj(qseq[:-1]),qseq[1:])
#   drpy = quat2rpy(dq)

#   t,integral_omega = quaternion.integrate_angular_velocity((dts,omega),dts[0],dts[-1],R0=qqseq[0])
  
def rpy_add_pi(rpy):
  return {'roll': mod2pi(rpy['roll']+np.pi), 'pitch': mod2pi(np.pi-rpy['pitch']), 'yaw': mod2pi(rpy['yaw']+np.pi)}

def quat2rpy_prior(q,rpyprev=None,epsilon=1e-15):
  rpy0 = quat2rpy(q)
  if rpyprev is None:
    rpyprev = {k: np.zeros_like(v) for k,v in rpy0.items()}
    
  isgimbal = np.abs(np.abs(rpy0['pitch'])-np.pi/2) < epsilon
  rpy1 = rpy_add_pi(rpy0)
  err0 = np.linalg.norm(np.array(list(rpy0.values()))-np.array(list(rpyprev.values())))
  err1 = np.linalg.norm(np.array(list(rpy1.values()))-np.array(list(rpyprev.values())))
  rpy = rpy1
  for k in rpy.keys():
    rpy[k][err0<err1] = rpy0[k][err0<err1]
    
  if np.any(isgimbal):
    # r, y
    # need delta to be dry
    if type(rpyprev['roll']) is np.ndarray:
      pass
    elif type(rpyprev['roll']) is float:
      rpyprev = {k: np.array([v,]) for k,v in rpyprev.items()}
    else:
      rpyprev = {k: np.array(v) for k,v in rpyprev.items()}
    if type(isgimbal) is not np.ndarray:
      isgimbal = np.array([isgimbal,])

    dryprev = rpyprev['roll'][isgimbal]-rpyprev['yaw'][isgimbal]
    dry = rpy0['roll'][isgimbal]-rpy0['yaw'][isgimbal]
    dd = dry-dryprev
    rpy['roll'][isgimbal] = rpyprev['roll'][isgimbal]+dd/2
    rpy['yaw'][isgimbal] = rpyprev['yaw'][isgimbal]-dd/2
    rpy['pitch'][isgimbal] = rpy0['pitch'][isgimbal]
  
  qrecon = rpy2quat(rpy['roll'],rpy['pitch'],rpy['yaw'])
  err = np.minimum(np.linalg.norm(qrecon-q,axis=-1),np.linalg.norm(-qrecon-q,axis=-1))
  
  return rpy,err,isgimbal
    

def transform_wing_angles(angles,body_pitch_angle=47.5):

  assert (WINGISTRANSFORMED==False)
  body_pitch_angle = np.deg2rad(body_pitch_angle)
  # Yaw: doesn't require transformation.
  # Roll.
  for i in range(angles.shape[-1]//3):
    off = i*3
    
    # stroke doesn't require transformation
    
    # deviation
    angles[:, off+angleidx['deviation']] = - angles[:, off+angleidx['deviation']]

    # rotation
    angles[:, off+angleidx['rotation']] = np.pi/2 - body_pitch_angle - angles[:, off+angleidx['rotation']]

  return angles

def get_downstroke(angles):
  
  # maxima and minima indices for downstrokes.
  starts = local_maxima_indices(angles[:, 0])
  ends = local_maxima_indices(-angles[:, 0])
  
  return starts,ends
  
def plot_wing_angles(angles, linewidth=1.5, transform_model_to_data=False, dt=0.0002):
    """
    Args:
        angles: (time, 6), the order: 
            yaw_left, roll_left, yaw_left,
            yaw_right, roll_right, yaw_right.
        transform_model_to_data (bool): Whether to transform `angles` from
            model joint definition to conventional fly angle definition.
    """
    angles = np.array(angles)
    if transform_model_to_data:
        body_pitch_angle = 47.5
        angles = transform_wing_angles(angles,body_pitch_angle)
    angles = np.rad2deg(angles)

    color_left = 'C0'
    color_right = 'C3'
    color_xaxis = [.7,.7,.7]
    downstroke_alpha = 0.09
    fs_labels = 16
    fs_ticks = 12
    
    factor = 1000  # Scale factor for x-axis. 1000 means ms, 1 means s.
    x_axis = np.linspace(0, factor*angles.shape[0]*dt, angles.shape[0])
    # Get maxima and minima indices for downstroke rectangles.
    maxima,minima = get_downstroke(angles)
    if minima[0] < maxima[0]:
        maxima = [0] + maxima
    
    fig = plt.figure(figsize=(14, 5))
    plt.subplots_adjust(hspace=.04)  # space between rows
    # plt.subplots_adjust(wspace=.03)  # space between columns
    axs = []

    # == Yaw.
    ax = plt.subplot(3, 1, 1)
    axs.append(ax)
    plt.plot([x_axis[0],x_axis[-1]],[0,0],'-',color=color_xaxis)
    plt.plot(x_axis, angles[:, 3], color_right, linewidth=linewidth,label='Right')
    plt.plot(x_axis, angles[:, 0], color_left, linewidth=linewidth,label='Left')
    plt.legend()
    # Downstrokes.
    ylim = plt.ylim()
    for maximum, minimum in zip(maxima, minima):
        ax.add_patch(Rectangle((factor*dt*maximum, -100), factor*dt*(minimum-maximum), 200, color='k', alpha=downstroke_alpha))
    plt.ylim(ylim)
    plt.ylabel(f'{wingangle_rename["yaw"]}\n(deg)', fontsize=fs_labels)
    plt.yticks(fontsize=fs_ticks)
    # == Roll.
    ax = plt.subplot(3, 1, 2,sharex=ax)
    axs.append(ax)
    plt.plot([x_axis[0],x_axis[-1]],[0,0],'-',color=color_xaxis)
    plt.plot(x_axis, angles[:, 4], color_right, linewidth=linewidth,label='Right')
    plt.plot(x_axis, angles[:, 1], color_left, linewidth=linewidth,label='Left')
    ylim = plt.ylim()
    # Downstrokes.
    for maximum, minimum in zip(maxima, minima):
        ax.add_patch(Rectangle((factor*dt*maximum, -100), factor*dt*(minimum-maximum), 200, color='k', alpha=downstroke_alpha))
    plt.ylim(ylim)
    plt.ylabel(f'{wingangle_rename["roll"]}\n(deg)', fontsize=fs_labels)
    plt.yticks(fontsize=fs_ticks)
    # == Pitch.
    ax = plt.subplot(3, 1, 3,sharex=ax)
    axs.append(ax)
    plt.plot([x_axis[0],x_axis[-1]],[0,0],'-',color=color_xaxis)
    plt.plot(x_axis, angles[:, 5], color_right, linewidth=linewidth,label='Right')
    plt.plot(x_axis, angles[:, 2], color_left, linewidth=linewidth,label='Left')
    ylim = plt.ylim()
    # Downstrokes.
    for maximum, minimum in zip(maxima, minima):
        ax.add_patch(Rectangle((factor*dt*maximum, -100), factor*dt*(minimum-maximum), 200, color='k', alpha=downstroke_alpha))
    plt.ylim(ylim)
    plt.ylabel(f'{wingangle_rename["pitch"]}\n(deg)', fontsize=fs_labels)
    plt.xlabel('Time (ms)', fontsize=fs_labels)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    
    return fig,np.array(axs)

def compute_wingangle_stats_over_strokes(wingangles,strokets,nsample=181):
  
  samplephases = np.linspace(0,1,nsample)
  ntraj = len(wingangles)
  nstrokes = np.sum([x.shape[0] for x in strokets])
  wingangles_per_stroke = np.zeros((nstrokes,nsample))
  off = 0
  for traji in range(ntraj):
    for strokei in range(strokets[traji].shape[0]):
      t0 = strokets[traji][strokei,0]
      t1 = strokets[traji][strokei,1]
      winganglecurr = wingangles[traji][t0:t1+1]
      phasescurr = np.linspace(0,1,t1-t0+1)
      winganglecurr_phase = np.interp(samplephases,phasescurr,winganglecurr)
      wingangles_per_stroke[off,:] = winganglecurr_phase
      off += 1
  meanwingangle = np.nanmean(wingangles_per_stroke,axis=0)
  stdwingangle = np.nanstd(wingangles_per_stroke,axis=0)
  return meanwingangle,stdwingangle,wingangles_per_stroke
    
def plot_attackangle_by_stroketype(modeldata,plotall=False):
  """
  plot_attackangle_by_stroketype(modeldata,plotall=False)
  Plot attack angle for left and right wings for downstroke and upstroke.
  If plotall==True, all strokes are plotted. Otherwise, standard deviation is plotted.
  Returns:
  fig, ax
  """
  sideoff = {'left': 0, 'right': 3}
  stroketypes = ['down','up']
  datatype_suffixes = {'real': '_ref', 'model': ''}
  meanattackangle = {}
  stdattackangle = {}
  attackangles_per_stroke = {}

  for datatype,suffix in datatype_suffixes.items():
    for side,off in sideoff.items():
      attackangles = [x[:,off+angleidx['rotation']] for x in modeldata['wing_qpos'+suffix]]
      for stroketype in stroketypes:
        fn = f'{datatype}_{side}_{stroketype}'
        meanattackangle[fn],stdattackangle[fn],attackangles_per_stroke[fn] = \
          compute_wingangle_stats_over_strokes(attackangles,modeldata[stroketype+'strokes'+suffix])

  cmcurr = matplotlib.colormaps.get_cmap('tab10')
  colors = {}
  for i,datatype in enumerate(datatype_suffixes.keys()):
    colors[datatype] = np.array(cmcurr(i))

  nsample = list(meanattackangle.values())[0].shape[0]
  phase = np.linspace(0,np.pi,nsample)
  ylim = [-np.pi/2,np.pi/2]
  
  fig,ax = plt.subplots(2,2,sharex=True,sharey=True)
  for i,side in enumerate(sideoff.keys()):
    for j,stroketype in enumerate(stroketypes):
      for k,datatype in enumerate(datatype_suffixes.keys()):
        fn = f'{datatype}_{side}_{stroketype}'
        if plotall:
          ax[j,i].plot(phase,attackangles_per_stroke[fn].T,color=colors[datatype][:3]*.5+.5,lw=.25,alpha=.25)
        else:
          ax[j,i].fill_between(phase,meanattackangle[fn]-stdattackangle[fn],
                              meanattackangle[fn]+stdattackangle[fn],color=colors[datatype][:3],alpha=.5,
                              linestyle='None')
      for k,datatype in enumerate(datatype_suffixes.keys()):
        fn = f'{datatype}_{side}_{stroketype}'
        ax[j,i].plot(phase,meanattackangle[fn],color=colors[datatype],lw=2,label=datatype)
      ax[j,i].set_title(f'{stroketype}stroke, {side} wing')
      ax[j,i].set_ylim(ylim)
  ax[-1,0].set_ylabel('Wing rotation angle')
  ax[-1,0].set_xlabel('Phase')
  # legend with location bottom right
  ax[0,0].legend(loc='lower right')
  fig.tight_layout()

  return fig,ax
  

def local_maxima_indices(x, consistency=1, proximity_threshold=None):
    """Returns indices of local maxima in x."""
    # Find local maxima.
    inds = []
    for i in range(consistency, len(x)-consistency):
        if all(x[i-consistency:i] < x[i]) and all(x[i] > x[i+1:i+consistency+1]):
            inds.append(i)
    # Maybe average over multiple maxima within proximity_threshold.
    if proximity_threshold is not None:
        inds_clean = []
        for i in inds:
            proximal_indices = []
            for j in inds:
                if abs(i - j) < proximity_threshold:
                    proximal_indices.append(j)
            inds_clean.append(int(np.mean(proximal_indices)))
        inds = sorted(list(set(inds_clean)))  # Remove duplicates.
    return inds

def load_data(datafile):
  # load in data from pickle file
  with open(datafile, 'rb') as f:
    data = pickle.load(f)
  print(data.keys())
  # qpos is a list of trajectories, each entry is of shape (time, 7)
  # qpos[:, :3] is the CoM and qpos[:, 3:7] is the orientation quaternion
  # wing_qpos  is the corresponding list of wing angles, each entry is (time, 6)
  # wing_qpos[:, :3] is left wing, wing_qpos[:, 3:] is right wing
  # For each wing, the order is yaw, roll, pitch
    
  # make the names match the reference data
  if 'qpos_model' in data:
    data['qpos'] = data['qpos_model']
    del data['qpos_model']
  if 'wings_model' in data:
    data['wing_qpos'] = data['wings_model']
    del data['wings_model']
    
  return data

def add_rpy_to_data(data,suffix=''):
  # add roll, pitch, yaw to the data
  data['roll'+suffix] = []
  data['pitch'+suffix] = []
  data['yaw'+suffix] = []
  for k in range(len(data['qpos'+suffix])):
    q = data['qpos'+suffix][k][:,3:]
    rpy = quatseq2rpy(q)
    for key in rpy.keys():
      data[key+suffix].append(rpy[key])
  return


def choose_omega_sigma(data,k=0,j=0):
  plt.clf()
  plt.plot(data['qpos_ref'][k][:,3+j],label='real')
  plt.plot(data['qpos'][k][:,3+j],label='unfiltered')  
  q0 = data['qpos'][k][:,3:]
  for sigma in range(1,10):
    q = q0.copy()
    q = ndimage.gaussian_filter1d(q, sigma=sigma, axis=0, order=0, mode='nearest',cval=np.nan)
    q = q/np.linalg.norm(q,axis=-1,keepdims=True)
    plt.plot(q[:,j],label=f'sigma={sigma}')
  plt.legend()
  return

def add_omega_to_data(data,dt,sigma=1,suffix='',use_rotated_axes=False):
  if use_rotated_axes:
    theta = np.deg2rad(-47.5)
  else:
    theta = None
  data['omega'+suffix] = []
  for k in range(len(data['qpos'+suffix])):
    q = data['qpos'+suffix][k][:,3:]
    if sigma is not None:
      q = ndimage.gaussian_filter1d(q, sigma=sigma, axis=0, order=0, mode='nearest')
      q = q/np.linalg.norm(q,axis=-1,keepdims=True)
    omega = quatseq2angvel(q,dt,theta=theta)
    # q = quaternion.as_quat_array(q)
    # dts = np.arange(len(q))*dt
    # omega = quaternion.angular_velocity(q,dts)
    data['omega'+suffix].append(omega)
  return

def add_vel_acc_to_data(data,dt,sigma=1,suffix=''):
  data['com_vel'+suffix] = []
  data['com_acc'+suffix] = []
  data['com_velmag'+suffix] = []
  data['com_accmag'+suffix] = []
  for i in range(len(data['qpos'+suffix])):
    com_pos = data['qpos'+suffix][i][:,:3]
    com_pos = ndimage.gaussian_filter1d(com_pos, sigma=sigma, axis=0, order=0, mode='constant',cval=np.nan)
    #com_vel = ndimage.gaussian_filter1d(com_pos, sigma=sigma, axis=0, order=1, mode='constant',cval=np.nan)/dt
    com_vel = np.gradient(com_pos,axis=0)/dt
    #com_vel = (com_pos[1:]-com_pos[:-1])/dt
    com_velmag = np.linalg.norm(com_vel,axis=1)
    #com_acc = ndimage.gaussian_filter1d(com_vel, sigma=sigma, axis=0, order=1, mode='constant',cval=np.nan)/dt
    #com_acc = np.gradient(com_vel,axis=0)/dt
    com_acc = (com_vel[1:]-com_vel[:-1])/dt
    com_accmag = np.linalg.norm(com_acc,axis=1)
    data['com_vel'+suffix].append(com_vel)
    data['com_acc'+suffix].append(com_acc)
    data['com_velmag'+suffix].append(com_velmag)
    data['com_accmag'+suffix].append(com_accmag)
  return

def plot_body_rpy_traj(modeldata,dt,nplot=10,idxplot=None):
  # plot roll pitch yaw for first nplot trajectories
  if idxplot is None:
    idxplot = np.arange(nplot,dtype=int)
  else:
    nplot = len(idxplot)
  fig,ax = plt.subplots(3,nplot,sharey='row',figsize=(30,15))
  keys = ['roll','pitch','yaw']
  for j in range(nplot):
    traji = idxplot[j]
    realts = np.arange(len(modeldata['roll_ref'][traji]))*dt
    modelts = np.arange(len(modeldata['roll'][traji]))*dt
    responset = modeldata['response_time'][traji]
    for i,key in enumerate(keys):
      ax[i,j].plot(realts,modeldata[key+'_ref'][traji],label='real')
      ax[i,j].plot(modelts,modeldata[key][traji],label='model')
      ax[i,j].plot(realts[responset],modeldata[key+'_ref'][traji][responset],'ko',label='response start')
      if j == 0:
        ax[i,j].set_ylabel(key)
    ax[0,j].set_title(f'Traj {traji}, turn {modeldata["turnangle_ref"][traji]*180/np.pi:.1f}')
  ax[0,0].legend()
  ax[2,0].set_xlabel('Time (s)')
  fig.tight_layout()
  return fig,ax

def plot_body_omega_traj(modeldata,dt,nplot=10,idxplot=None):
  if idxplot is None:
    idxplot = np.arange(nplot,dtype=int)
  else:
    nplot = len(idxplot)
  fig,ax = plt.subplots(3,nplot,sharey='row',figsize=(30,15))
  keys = ['x','y','z']
  for j in range(nplot):
    traji = idxplot[j]
    realts = np.arange(modeldata['omega_ref'][traji].shape[0])*dt
    modelts = np.arange(modeldata['omega'][traji].shape[0])*dt
    responset = modeldata['response_time'][traji]
    for i,key in enumerate(keys):
      ax[i,j].plot(realts,modeldata['omega_ref'][traji][...,i],label='real')
      ax[i,j].plot(modelts,modeldata['omega'][traji][...,i],label='model')
      ax[i,j].plot(realts[responset],modeldata['omega_ref'][traji][responset,i],'ko',label='response start')
      if j == 0:
        ax[i,j].set_ylabel(f'omega {key} (rad/s)')
    ax[0,j].set_title(f'Traj {traji}, turn {modeldata["turnangle"][traji]*180/np.pi:.1f}')
  ax[0,0].legend()
  ax[2,0].set_xlabel('Time (s)')
  fig.tight_layout()
  return fig,ax

def turnangle2str(angle):
  angle = int(angle)
  if angle < 0:
    anglestr = f'{-angle}L'
  elif angle > 0:
    anglestr = f'{angle}R'
  else:
    anglestr = '0'
  return anglestr

def plot_compare_omega_response(modeldata,*args,**kwargs):
  nplot = 4
  fig,ax = plt.subplots(nplot,2,sharex=True,sharey='row',figsize=(10,10))
  plot_omega_response(modeldata,*args,ax=ax[:,0],**kwargs,suffix='_ref')
  plot_omega_response(modeldata,*args,ax=ax[:,1],**kwargs,suffix='')  
  ax[0,0].legend()
  ax[0,0].set_title('Real')
  ax[0,1].set_title('Model')
  fig.tight_layout()
  return fig,ax

def plot_omega_response(modeldata,dt,deltaturn=200,plotall=True,nbins=4,
                        ax=None,minturnangle=-180,maxturnangle=0,suffix=''):
  
  # choose bin edges
  angleturn_bin_edges = np.linspace(minturnangle,maxturnangle,nbins+1)
  angleturn_bin_edges[-1]+=1e-6
  binnames = []
  for i in range(nbins):
    angle0 = turnangle2str(angleturn_bin_edges[i])
    angle1 = turnangle2str(angleturn_bin_edges[i+1])
    binnames.append(f'[{angle0},{angle1})')
      
  ntraj = len(modeldata['omega'+suffix])
  T = deltaturn*2+1
  nplot = 4
  plotnames = ['$\omega_x$ (deg/s)','$\omega_y$ (deg/s)','$\omega_z$ (deg/s)','Vel heading (deg)']
  dataplot = np.zeros((ntraj,T,nplot))
  
  def alignfun(x,t0,t1):
    ndim = x.ndim
    x = np.pad(x,[(deltaturn,deltaturn),]+[(0,0),]*(ndim-1),mode='constant',constant_values=np.nan)
    x = x[t0:t1+1]
    talign = np.nonzero(np.isnan(x)==False)[0][0]
    x -= x[talign]
    return x
  
  for i in range(ntraj):
    if np.abs(modeldata['omega'][i].shape[0]-modeldata['omega_ref'][i].shape[0]) > 10:
      print(f'Warning: traj {i} has different lengths for model and real data')
      continue
    tresponse = modeldata['response_time'][i]
    t = tresponse+deltaturn
    t0 = t-deltaturn
    t1 = t+deltaturn
    assert (np.isnan(modeldata['veldir_xy'][i][tresponse])==False)
    assert (np.isnan(modeldata['veldir_xy_ref'][i][tresponse])==False)
    dataplot[i,:,:3] = alignfun(modeldata['omega'+suffix][i],t0,t1)*180/np.pi
    dataplot[i,:,3] = alignfun(modeldata['veldir_xy'+suffix][i],t0,t1)*180/np.pi

  # flip angles so they are all left turns
  turnangle_xy = modeldata['turnangle_xy'+suffix]*180/np.pi
  idxpositive = turnangle_xy > 0
  turnangle_xy[idxpositive] = -turnangle_xy[idxpositive]
  # flip omega_x, omega_z, veldir
  for i in [0,2,3]:
    dataplot[idxpositive,:,i] = -dataplot[idxpositive,:,i]

  meandataplot = np.zeros((nbins,T,nplot))
  stderr_dataplot = np.zeros((nbins,T,nplot))
  for i in range(nbins):
    idx = (turnangle_xy >= angleturn_bin_edges[i]) & (turnangle_xy < angleturn_bin_edges[i+1])
    # omega should be near 0, so taking averages should be fine
    # veldir_xy angles have been unwrapped, so 360 offsets should be meaningful. starts at 0
    meandataplot[i] = np.nanmean(dataplot[idx],axis=0)
    n = np.sum(np.any(np.isnan(dataplot[idx]),axis=-1)==False,axis=0)
    stderr_dataplot[i] = np.nanstd(dataplot[idx],axis=0)/np.sqrt(n[...,None])

  if ax is None:
    createdfig = True
    fig,ax = plt.subplots(nplot,1,sharex=True,sharey='row',figsize=(10,10))
  else:
    createdfig = False
    
  cmcurr = matplotlib.colormaps.get_cmap('jet')
  bincenters = (angleturn_bin_edges[:-1]+angleturn_bin_edges[1:])/2
  meancolors = cmcurr((bincenters-minturnangle)/(maxturnangle-minturnangle))
  ts = np.arange(-deltaturn,deltaturn+1)*dt
  
  # plot shaded standard error
  for i in range(nbins):
    color = meancolors[i]*.5
    for j in range(nplot):
      ax[j].fill_between(ts,meandataplot[i,:,j]-stderr_dataplot[i,:,j],
                         meandataplot[i,:,j]+stderr_dataplot[i,:,j],color=color,alpha=.5,
                         linestyle='None')
  if plotall:
    for i in range(nbins):
      idx = (turnangle_xy >= angleturn_bin_edges[i]) & (turnangle_xy < angleturn_bin_edges[i+1])
      for k in np.nonzero(idx)[0]:
        for j in range(nplot):
          color = np.array(cmcurr((turnangle_xy[k]-minturnangle)/(maxturnangle-minturnangle))[:3])
          color = color*.5 + .25
          ax[j].plot(ts,dataplot[k,:,j].T,color=color,lw=.25)

  for j in range(nplot):
    for i in range(nbins):
      color = meancolors[i]*.5
      ax[j].plot(ts,meandataplot[i,:,j].T,color=color,lw=2)
    ax[j].set_ylabel(plotnames[j])
    ax[j].grid(visible=True,axis='x')

  ax[-1].set_xlabel('Time (s)')

  if createdfig:
    ax[0].legend()
    fig.tight_layout()
    
  if createdfig:
    return fig,ax
  else:
    return


def plot_rpy_response(modeldata,dt,deltaturn=200):
  nbins = 4
  minturnangle = -np.pi
  maxturnangle = 0
  angleturn_bin_edges = np.linspace(minturnangle,maxturnangle,nbins+1)
  angleturn_bin_edges[-1]+=1e-6
  binnames = []
  for i in range(nbins):
    angle0 = int(angleturn_bin_edges[i]*180/np.pi)
    angle1 = int(angleturn_bin_edges[i+1]*180/np.pi)
    if angle0 < 0:
      angle0str = f'{-angle0}L'
    elif angle0 > 0:
      angle0str = f'{angle0}R'
    else:
      angle0str = '0'
    if angle1 < 0:
      angle1str = f'{-angle1}L'
    elif angle1 > 0:
      angle1str = f'{angle1}R'
    else:
      angle1str = '0'
      
    binnames.append(f'[{angle0str},{angle1str})')
      
  ntraj = len(modeldata['roll'])
  T = deltaturn*2+1
  rpy = np.zeros((ntraj,T,3))
  rpy[:] = np.nan
  rpy_ref = np.zeros((ntraj,T,3))
  rpy_ref[:] = np.nan
  keys = ['roll','pitch','yaw']
  for i in range(ntraj):
    if np.abs(len(modeldata['roll'][i])-len(modeldata['roll_ref'][i])) > 10:
      print(f'Warning: traj {i} has different lengths for model and real data')
      continue
    tresponse = modeldata['response_time'][i]
    for j,key in enumerate(keys):
      x = modeldata[key][i]
      x = np.pad(x,deltaturn,mode='constant',constant_values=np.nan)
      t = tresponse+deltaturn
      t0 = t-deltaturn
      t1 = t+deltaturn
      x = x[t0:t1+1]-x[t0]
      rpy[i,:,j] = x
      x = modeldata[key+'_ref'][i]
      x = np.pad(x,deltaturn,mode='constant',constant_values=np.nan)
      x = x[t0:t1+1]-x[t0]
      rpy_ref[i,:,j] = x

  turnangle_xy = modeldata['turnangle_xy_ref'].copy()
  idxpositive = turnangle_xy > 0
  turnangle_xy[idxpositive] = -turnangle_xy[idxpositive]
  # flip roll and yaw
  for i in [0,2]:
    rpy_ref[idxpositive,:,i] = -rpy_ref[idxpositive,:,i]
    rpy[idxpositive,:,i] = -rpy[idxpositive,:,i]
  meanrpy = np.zeros((nbins,T,3))
  meanrpy_ref = np.zeros((nbins,T,3))
  stderr_rpy = np.zeros((nbins,T,3))
  stderr_rpy_ref = np.zeros((nbins,T,3))
  for i in range(nbins):
    idx = (turnangle_xy >= angleturn_bin_edges[i]) & (turnangle_xy < angleturn_bin_edges[i+1])
    meanrpy_ref[i] = np.nanmean(rpy_ref[idx],axis=0)
    meanrpy[i] = np.nanmean(rpy[idx],axis=0)
    n = np.sum(np.any(np.isnan(rpy[idx]),axis=-1)==False,axis=0)
    nref = np.sum(np.any(np.isnan(rpy_ref[idx]),axis=-1)==False,axis=0)
    stderr_rpy_ref[i] = np.nanstd(rpy_ref[idx],axis=0)/np.sqrt(nref[...,None])
    stderr_rpy[i] = np.nanstd(rpy[idx],axis=0)/np.sqrt(n[...,None])

  fig,ax = plt.subplots(3,2,sharex=True,sharey='row',figsize=(10,10))
  cmcurr = matplotlib.colormaps.get_cmap('jet')
  bincenters = (angleturn_bin_edges[:-1]+angleturn_bin_edges[1:])/2
  meancolors = cmcurr((bincenters-minturnangle)/(maxturnangle-minturnangle))
  ts = np.arange(-deltaturn,deltaturn+1)*dt
  
  # plot shaded standard error
  for i in range(nbins):
    color = meancolors[i]*.5
    for j in range(3):
      ax[j,0].fill_between(ts,meanrpy_ref[i,:,j]-stderr_rpy_ref[i,:,j],
                           meanrpy_ref[i,:,j]+stderr_rpy_ref[i,:,j],color=color,alpha=.5,
                           linestyle='None')
      ax[j,1].fill_between(ts,meanrpy[i,:,j]-stderr_rpy[i,:,j],
                           meanrpy[i,:,j]+stderr_rpy[i,:,j],color=color,alpha=.5,
                           linestyle='None')
  for i in range(nbins):
    idx = (turnangle_xy >= angleturn_bin_edges[i]) & (turnangle_xy < angleturn_bin_edges[i+1])
    for j in range(3):
      for k in np.nonzero(idx)[0]:
        color = np.array(cmcurr((turnangle_xy[k]-minturnangle)/(maxturnangle-minturnangle))[:3])
        color = color*.5 + .25
        ax[j,0].plot(ts,rpy_ref[k,:,j].T,color=color,lw=.25)
        ax[j,1].plot(ts,rpy[k,:,j].T,color=color,lw=.25)
      ax[j,0].set_ylabel(keys[j])

  for i in range(nbins):
    for j in range(3):
      color = meancolors[i]*.5
      ax[j,0].plot(ts,meanrpy_ref[i,:,j].T,color=color,lw=2,
                   label=binnames[i])
      ax[j,1].plot(ts,meanrpy[i,:,j].T,color=color,lw=2)
      ax[j,0].grid(visible=True,axis='x')
      ax[j,1].grid(visible=True,axis='x')

  
  ax[0,0].set_title('Real')
  ax[0,1].set_title('Model')
  ax[-1,0].set_xlabel('Time (s)')
  ax[0,0].legend()
  fig.tight_layout()
    
  return fig,ax

def add_response_times(data,accmag_thresh,minframe_max=100,doplot=False,suffix=''):
  ntraj = len(data['com_accmag'+suffix])
  nc = int(np.ceil(np.sqrt(ntraj)))
  nr = int(np.ceil(ntraj/nc))
  if doplot:
    fig,ax = plt.subplots(nr,nc,figsize=(20,30))
    ax = ax.flatten()
  data['response_time'] = np.zeros(ntraj,dtype=int)
  for i in range(ntraj):
    accmag = data['com_accmag'+suffix][i]
    i1 = np.nanargmax(accmag[minframe_max:])+minframe_max
    i0 = np.nonzero(accmag[:i1] <= accmag_thresh)[0][-1]
    data['response_time'][i] = i0
    if doplot:
      ax[i].plot(accmag)
      ax[i].plot([i0,i0],[0,2000])
  return

def add_veldir(data,suffix='',sigma=10,epsilon=1e-3,deltaturn=200):
  ntraj = len(data['com_vel'+suffix])
  data['veldir'+suffix] = []
  data['veldir_xy'+suffix] = []
  for i in range(ntraj):
    vel = data['com_vel'+suffix][i].copy()
    if sigma is not None:
      realidx = np.nonzero(np.isnan(vel)==False)[0]
      if realidx[0] > 0:
        vel[:realidx[0]] = vel[realidx[0]]
      if realidx[-1] < vel.shape[0]-1:
        vel[realidx[-1]+1:] = vel[realidx[-1]]      
      vel = ndimage.gaussian_filter1d(vel, sigma=sigma, axis=0, order=0, mode='nearest')
      vel[:realidx[0]-1] = np.nan
      vel[realidx[-1]+1:] = np.nan
      
    z = np.linalg.norm(vel,axis=-1,keepdims=True)
    z = np.maximum(epsilon,z)
    veldir = vel / z
    veldir_xy0 = np.arctan2(veldir[:,1],veldir[:,0])
    veldir_xy = unwrap_angleseq(veldir_xy0,t0=data['response_time'][i]-deltaturn)
    assert np.all(np.any(np.isnan(veldir),axis=-1) == np.isnan(veldir_xy))
    assert np.nanmax(np.abs(mod2pi(veldir_xy-veldir_xy0))) < 1e-3
    data['veldir'+suffix].append(veldir)
    data['veldir_xy'+suffix].append(veldir_xy)
  return  

def add_turnangles(data,deltaturn=200,suffix=''):
  ntraj = len(data['com_accmag'+suffix])
  data['turnangle'+suffix] = np.zeros(ntraj)
  data['turnangle_xy'+suffix] = np.zeros(ntraj)
  for i in range(ntraj):
    veldir = data['veldir'+suffix][i]
    veldir_xy = data['veldir_xy'+suffix][i]
    T = np.nonzero(np.all(np.isnan(veldir),axis=1)==False)[0][-1]
    t0 = data['response_time'][i]
    if t0 > T:
      print(f'Warning: traj {i} has response time {t0} > T {T}, using reference data to get turn angle')
      veldir = data['veldir_ref'][i]
      veldir_xy = data['veldir_xy_ref'][i]
      T = np.nonzero(np.any(np.isnan(veldir),axis=1)==False)[0][-1]
    t1 = np.clip(t0+deltaturn,0,T)
    turnangle = np.arccos(np.dot(veldir[t0],veldir[t1]))
    # no modpi because we unwrapped veldir_xy
    dangle_xy = veldir_xy[t1]-veldir_xy[t0]
    # choose sign based on sign of angle in x-y plane
    turnangle = np.sign(dangle_xy) * turnangle
    data['turnangle'+suffix][i] = turnangle
    data['turnangle_xy'+suffix][i] = dangle_xy
    assert (np.isnan(dangle_xy)==False)
    
  return

def filter_trajectories(data,trajidx):
  ntraj = len(data['qpos'])
  for k,v in data.items():
    if (type(v) == list) and len(v) == ntraj:
      print(f'filtering list {k}')
      data[k] = [v[i] for i in trajidx]
    elif (type(v) == np.ndarray) and v.shape[0] == ntraj:
      print(f'filtering ndarray {k}')
      data[k] = v[trajidx]
    else:
      print(f'skipping {k}')
  return

def add_wing_qpos_ref(realdata,modeldata):
  """
  add_wing_qpos_ref(realdata,modeldata)
  Add wing_qpos_ref for the real data to the model data. Swap left and right wings if the trajectory is flipped.
  Modifies modeldata in place. 
  """
  ntraj = len(modeldata['qpos'])
  modeldata['wing_qpos_ref'] = []
  
  for i in range(ntraj):
    reali = modeldata['traj_inds_orig'][i]
    isflipped = reali < 0
    if isflipped:
      reali = -reali
    wing_qpos = realdata['wing_qpos'][reali].copy()
    modeldata['wing_qpos_ref'].append(wing_qpos[:,[3,4,5,0,1,2]])
  return

def compare_rpy_matlab(realdata,rpydatafile,dt):
  """
  compare_rpy_matlab(realdata,rpydatafile,dt)
  Compare roll pitch yaw computed by matlab to that computed by python.
  Plots roll pitch and yaw computed in three ways for the first 10 trajectories.
  """
  
  matdata,_ = loadmat(rpydatafile)
  rpy_loaded = np.array(matdata['rpy_loaded'])
  rpy_loaded = np.transpose(rpy_loaded,(2,1,0))
  rpy_matcomputed = np.array(matdata['rpy_computed'])
  rpy_matcomputed = np.transpose(rpy_matcomputed,(2,1,0))
  dts_mat = np.array(matdata['dts'])
  dts_mat = dts_mat[:,0]
  
  nplot = 10
  fig,ax = plt.subplots(3,10,sharex='col',sharey=True,figsize=(30,15))
  keys = ['roll','pitch','yaw']
  for traji in range(nplot):
    for i in range(3):
      key = keys[i]
      js = np.nonzero(np.isnan(rpy_loaded[:,traji,i])==False)[0]
      j0 = js[0]
      j1 = js[-1]
      dts_mat_curr = dts_mat[j0:j1]-dts_mat[j0]
      dts_py_curr = np.arange(len(realdata[key][traji]))*dt
      ax[i,traji].plot(dts_mat_curr,rpy_loaded[j0:j1,traji,i],label='loaded')
      ax[i,traji].plot(dts_mat_curr,rpy_matcomputed[j0:j1,traji,i],label='matlab computed')
      ax[i,traji].plot(dts_py_curr,realdata[key][traji],'--',label='python computed')
  ax[0,0].legend()
  for i in range(3):
    ax[i,0].set_ylabel(keys[i])
  ax[-1,0].set_xlabel('Time (s)')
  fig.tight_layout()  
  
def plot_wing_angle_trajs(modeldata,idxplot,dosave=False):
  for traji in idxplot:
    figreal,axreal = plot_wing_angles(modeldata['wing_qpos_ref'][traji],transform_model_to_data=False)
    figmodel,axmodel = plot_wing_angles(modeldata['wing_qpos'][traji],transform_model_to_data=False)
    axmodel[0].set_title('Model')
    axreal[0].set_title('Real')
    # link axes for the real and model plots
    for i in range(len(axreal)):
      ylimreal = axreal[i].get_ylim()
      ylimmodel = axmodel[i].get_ylim()
      ylim = (min(ylimreal[0],ylimmodel[0]),max(ylimreal[1],ylimmodel[1]))
      axreal[i].set_ylim(ylim)
      axmodel[i].set_ylim(ylim)
    if dosave:
      figreal.savefig(f'real_wingangles_traj{traji}.png')
      figmodel.savefig(f'model_wingangles_traj{traji}.png')
      figreal.savefig(f'real_wingangles_traj{traji}.svg')
      figmodel.savefig(f'model_wingangles_traj{traji}.svg')
      plt.close(figreal)
      plt.close(figmodel)
      
def add_wingstroke_timing(data,suffix=''):
  
  body_pitch_angle = 47.5
  data['downstrokes'+suffix] = []
  data['upstrokes'+suffix] = []
  ntraj = len(data['wing_qpos'+suffix])
  for traji in range(ntraj):
    angles = data['wing_qpos'+suffix][traji]
    # downstrokes go from maxima to minima
    # upstrokes go from minima to maxima
    maxima,minima = get_downstroke(angles)
    if maxima[0] < minima[0]:
      ndownstrokes = np.minimum(len(minima),len(maxima))
      downstrokes = np.c_[maxima[:ndownstrokes],minima[:ndownstrokes]]
      nupstrokes = np.minimum(len(minima)-1,len(maxima))
      upstrokes = np.c_[minima[:nupstrokes],maxima[1:nupstrokes+1]]
    else:
      ndownstrokes = np.minimum(len(minima)-1,len(maxima))
      downstrokes = np.c_[maxima[:ndownstrokes],minima[1:ndownstrokes+1]]
      nupstrokes = np.minimum(len(minima),len(maxima))
      upstrokes = np.c_[minima[:nupstrokes],maxima[:nupstrokes]]
    data['downstrokes'+suffix].append(downstrokes)
    data['upstrokes'+suffix].append(upstrokes)
  
  return

# def compute_acceleration_stats(realdata):
#   mean_com_absacc = np.nanmean(np.abs(np.concatenate(realdata['com_acc'],axis=0)),axis=0)
#   mean_com_absacc = np.nanmean(np.abs(np.concatenate(realdata['com_acc'],axis=0)),axis=0)
#   realdata['com_acc']
#   return

# def plot_stroke_variability(modeldata):
  
#   return
savefigformats = ['png','svg']
def mysavefig(fig,name):
  for fmt in savefigformats:
    filename = f'{name}.{fmt}'
    print(f'Saving figure to {filename}')
    fig.savefig(filename)

if __name__ == "__main__":
  print('Hello!')

  accmag_thresh = .28 * 9.81 * 1e2 # 0.28 g  
  sigma_smooth = 8 # chosen by eye so that wiggles in omega for model data look like wiggles in omega for real data
  #testtrajidx = np.array([  5,   6,   8,  13,  14,  19,  22,  23,  24,  32,  43,  57,  59, 68,  75,  79,  85,  87,  90, 101, 103, 114, 117, 118, 120, 131, 132, 134, 137, 138, 148, 150, 153, 157, 173, 183, 185, 190, 194, 204, 205, 211, 213, 214, 216, 223, 226, 233, 241, 246, 258, 266, 268, 270, 271])
  testtrajidx = np.array([  5,   6,   8,  13,  14,  19,  22,  23,  24,  32,  43,  57,  59, 68,  75,  79,  85,  87,  90, 101, 103, 114, 117, 118, 120, 131, 132, 134, 137, 138, 148, 150, 153, 157, 173, 183, 185, 190, 194, 204, 205, 211, 213, 214, 216, 223, 226, 233, 241, 246, 258, 266, 268, 270, 271])
  realdatafile = 'flight-dataset_wing-qpos_not-augmented_evasion-saccade_n-traj-136.pkl'
  modeldatafile = 'analysis-rollouts-272_flight-imitation-wbpg-shaping_no-root-quat-no-height-obs_init-rot-qvel_data-hdf5_start-step-random_net-1_wing-prms-18_split-train-test_margin-0.4-pi_seed-1.pkl'
  rpydatafile = 'rpy.mat'
  realdata = load_data(realdatafile)
  allmodeldata = load_data(modeldatafile)
  savefigs = True

  # which non-flipped example does each model trajectory come from?
  ntraj_per_dataset = [92,44]
  allmodeldata['traj_inds_orig'] = []
  off = 0
  for n in ntraj_per_dataset:
    # list from 0 to n-1
    allmodeldata['traj_inds_orig'] += list(range(off,off+n))
    allmodeldata['traj_inds_orig'] += [-x for x in range(off,off+n)]
    off += n
  add_wing_qpos_ref(realdata,allmodeldata)
  
  # transform wing angle -- don't do this more than once!!
  assert WINGISTRANSFORMED == False
  for i in range(len(allmodeldata['wing_qpos'])):
    allmodeldata['wing_qpos_ref'][i] = transform_wing_angles(allmodeldata['wing_qpos_ref'][i])
    allmodeldata['wing_qpos'][i] = transform_wing_angles(allmodeldata['wing_qpos'][i])
  for i in range(len(realdata['wing_qpos'])):
    realdata['wing_qpos'][i] = transform_wing_angles(realdata['wing_qpos'][i])
  WINGISTRANSFORMED = True
  
  dt = realdata['dt']
  deltaturn = int(40e-3/dt) # frame

  # compute angular velocity
  add_omega_to_data(allmodeldata,dt,sigma=sigma_smooth)
  add_omega_to_data(allmodeldata,dt,suffix='_ref',sigma=None)
  add_omega_to_data(realdata,dt,sigma=None)

  # compute roll pitch yaw
  add_rpy_to_data(allmodeldata)
  add_rpy_to_data(allmodeldata,suffix='_ref')
  add_rpy_to_data(realdata)
  
  compare_rpy_matlab(realdata,rpydatafile,dt)  
  
  # compute vel and acc
  add_vel_acc_to_data(allmodeldata,dt)
  add_vel_acc_to_data(allmodeldata,dt,suffix='_ref')
  add_vel_acc_to_data(realdata,dt)
  
  # compute response times
  add_response_times(allmodeldata,accmag_thresh,suffix='_ref')
  add_response_times(realdata,accmag_thresh)

  # add velocity direction info
  add_veldir(allmodeldata,sigma=sigma_smooth)
  add_veldir(allmodeldata,sigma=None,suffix='_ref')
  add_veldir(realdata,sigma=None)

  # compute turn angles
  add_turnangles(allmodeldata,suffix='_ref',deltaturn=deltaturn)
  add_turnangles(allmodeldata,deltaturn=deltaturn)
  add_turnangles(realdata,deltaturn=deltaturn)
  
  add_wingstroke_timing(allmodeldata,suffix='_ref')
  add_wingstroke_timing(allmodeldata)
  add_wingstroke_timing(realdata)

  
  modeldata = copy.deepcopy(allmodeldata)
  filter_trajectories(modeldata,testtrajidx)  
  
  #fig,ax = plot_rpy_response(modeldata,dt,deltaturn=deltaturn)
  fig,ax = plot_compare_omega_response(modeldata,dt,deltaturn=deltaturn,plotall=False)
  if savefigs:
    mysavefig(fig,'omega_response')
  fig,ax = plot_compare_omega_response(allmodeldata,dt,deltaturn=deltaturn,plotall=False)
  if savefigs:
    mysavefig(fig,'omega_response_all')
  fig,ax = plot_compare_omega_response(modeldata,dt,deltaturn=deltaturn,plotall=False,nbins=1)
  if savefigs:
    mysavefig(fig,'omega_response_notbinned')

  ntraj = len(modeldata['qpos'])

  # jebdata = copy.deepcopy(allmodeldata)
  # jebtrajidx = np.nonzero(np.abs(allmodeldata['traj_inds_orig'])>=ntraj_per_dataset[0])[0]
  # filter_trajectories(allmodeldata,jebtrajidx)
  
  order = np.argsort(modeldata['turnangle_ref'])
  idxplot = np.r_[order[:5],order[-5:]]

  # plot roll pitch yaw for nplot trajectories
  fig,ax = plot_body_rpy_traj(modeldata,dt,idxplot=idxplot)
  fig,ax = plot_body_omega_traj(modeldata,dt,idxplot=idxplot)
    
  # plot wing angles
  # if savefigs:
  #   plot_wing_angle_trajs(modeldata,idxplot,dosave=True)  
  plot_wing_angle_trajs(modeldata,idxplot[[0,]])
  
  fig,ax = plot_attackangle_by_stroketype(modeldata,plotall=True)
  if savefigs:
    mysavefig(fig,'attackangle')
  print('Goodbye!')