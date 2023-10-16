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
  roll  = np.arctan2(2.0 * (q[:,zidx] * q[:,yidx] + q[:,widx] * q[:,xidx]) , 1.0 - 2.0 * (q[:,xidx] * q[:,xidx] + q[:,yidx] * q[:,yidx]))
  pitch = np.arcsin(2.0 * (q[:,yidx] * q[:,widx] - q[:,zidx] * q[:,xidx]))
  yaw   = np.arctan2(2.0 * (q[:,zidx] * q[:,widx] + q[:,xidx] * q[:,yidx]) , - 1.0 + 2.0 * (q[:,widx] * q[:,widx] + q[:,xidx] * q[:,xidx]))
  return {'roll': roll, 'pitch': pitch, 'yaw': yaw}

def quatconj(q):
  return np.c_[q[:,0],-q[:,1],-q[:,2],-q[:,3]]

def quatmultiply(q,r):
  n0 = r[:,0]*q[:,0] - r[:,1]*q[:,1] - r[:,2]*q[:,2] - r[:,3]*q[:,3]
  n1 = r[:,0]*q[:,1] + r[:,1]*q[:,0] - r[:,2]*q[:,3] + r[:,3]*q[:,2]
  n2 = r[:,0]*q[:,2] + r[:,1]*q[:,3] + r[:,2]*q[:,0] - r[:,3]*q[:,1]
  n3 = r[:,0]*q[:,3] - r[:,1]*q[:,2] + r[:,2]*q[:,1] + r[:,3]*q[:,0]
  n = np.c_[n0,n1,n2,n3]
  z = np.linalg.norm(n,axis=1)
  n = n / z[:,None]
  return n

def quatseq2rpy(q):
  dq = quatmultiply(quatconj(q[:-1]),q[1:])
  drpy = quat2rpy(dq)
  sz = drpy['roll'].shape
  z = np.zeros((1,)+sz[1:])
  rpy = {}
  for k,v in drpy.items():
    rpy[k] = np.cumsum(np.r_[z,v],axis=0)
  return rpy

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

def compute_attackangle_stats(attackangles,strokets,nsample=181):
  
  samplephases = np.linspace(0,1,nsample)
  ntraj = len(attackangles)
  nstrokes = np.sum([x.shape[0] for x in strokets])
  attackangles_per_stroke = np.zeros((nstrokes,nsample))
  off = 0
  for traji in range(ntraj):
    for strokei in range(strokets[traji].shape[0]):
      t0 = strokets[traji][strokei,0]
      t1 = strokets[traji][strokei,1]
      attackcurr = attackangles[traji][t0:t1+1]
      phasescurr = np.linspace(0,1,t1-t0+1)
      attackcurr_phase = np.interp(samplephases,phasescurr,attackcurr)
      attackangles_per_stroke[off,:] = attackcurr_phase
      off += 1
  meanattackangle = np.nanmean(attackangles_per_stroke,axis=0)
  stdattackangle = np.nanstd(attackangles_per_stroke,axis=0)
  return meanattackangle,stdattackangle,attackangles_per_stroke
    
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
          compute_attackangle_stats(attackangles,modeldata[stroketype+'strokes'+suffix])

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
  for k in range(len(data['qpos'])):
    q = data['qpos'+suffix][k][:,3:]
    rpy = quatseq2rpy(q)
    for key in rpy.keys():
      data[key+suffix].append(rpy[key])
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
    ax[0,j].set_title(f'Traj {traji}, turn {modeldata["turnangle"][traji]*180/np.pi:.1f}')
  ax[0,0].legend()
  ax[2,0].set_xlabel('Time (s)')
  fig.tight_layout()
  return fig,ax

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

  turnangle_xy = modeldata['turnangle_xy'].copy()
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

def add_turnangles(data,deltaturn=200,suffix=''):
  ntraj = len(data['com_accmag'+suffix])
  data['turnangle'] = np.zeros(ntraj)
  data['turnangle_xy'] = np.zeros(ntraj)
  for i in range(ntraj):
    vel = data['com_vel'+suffix][i]
    T = np.nonzero(np.any(np.isnan(vel),axis=1)==False)[0][-1]
    t0 = data['response_time'][i]
    t1 = np.minimum(t0+deltaturn,T)
    vel0 = vel[t0]
    vel1 = vel[t1]
    veldir0 = vel0 / np.linalg.norm(vel0)
    veldir1 = vel1 / np.linalg.norm(vel1)
    turnangle = np.arccos(np.dot(veldir0,veldir1))
    # choose sign based on sign of angle in x-y plane
    dangle_xy = np.arctan2(veldir1[1],veldir1[0]) - np.arctan2(veldir0[1],veldir0[0])
    turnangle = np.sign(dangle_xy) * turnangle
    data['turnangle'][i] = turnangle
    data['turnangle_xy'][i] = dangle_xy
    
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
    angles = modeldata['wing_qpos'+suffix][traji]
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

if __name__ == "__main__":
  print('Hello!')

  accmag_thresh = .28 * 9.81 * 1e2 # 0.28 g  
  #testtrajidx = np.array([  5,   6,   8,  13,  14,  19,  22,  23,  24,  32,  43,  57,  59, 68,  75,  79,  85,  87,  90, 101, 103, 114, 117, 118, 120, 131, 132, 134, 137, 138, 148, 150, 153, 157, 173, 183, 185, 190, 194, 204, 205, 211, 213, 214, 216, 223, 226, 233, 241, 246, 258, 266, 268, 270, 271])
  testtrajidx = np.array([  5,   6,   8,  13,  14,  19,  22,  23,  24,  32,  43,  57,  59, 68,  75,  79,  85,  87,  90, 101, 103, 114, 117, 118, 120, 131, 132, 134, 137, 138, 148, 150, 153, 157, 173, 183, 185, 190, 194, 204, 205, 211, 213, 214, 216, 223, 226, 233, 241, 246, 258, 266, 268, 270, 271])
  realdatafile = 'flight-dataset_wing-qpos_not-augmented_evasion-saccade_n-traj-136.pkl'
  modeldatafile = 'analysis-rollouts-272_flight-imitation-wbpg-shaping_no-root-quat-no-height-obs_init-rot-qvel_data-hdf5_start-step-random_net-1_wing-prms-18_split-train-test_margin-0.4-pi_seed-1.pkl'
  rpydatafile = 'rpy.mat'
  realdata = load_data(realdatafile)
  allmodeldata = load_data(modeldatafile)

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
  
  modeldata = copy.deepcopy(allmodeldata)
  filter_trajectories(modeldata,testtrajidx)
    
  dt = realdata['dt']
  deltaturn = int(40e-3/dt) # frame

  # compute roll pitch yaw
  add_rpy_to_data(modeldata)
  add_rpy_to_data(modeldata,suffix='_ref')
  add_rpy_to_data(realdata)
  
  compare_rpy_matlab(realdata,rpydatafile,dt)  
  
  # compute vel and acc
  add_vel_acc_to_data(modeldata,dt)
  add_vel_acc_to_data(modeldata,dt,suffix='_ref')
  
  # compute response times
  add_response_times(modeldata,accmag_thresh,suffix='_ref')

  # compute turn angles
  add_turnangles(modeldata,suffix='_ref',deltaturn=deltaturn)
  
  fig,ax = plot_rpy_response(modeldata,dt,deltaturn=deltaturn)
  ntraj = len(modeldata['qpos'])
  
  order = np.argsort(modeldata['turnangle'])
  idxplot = np.r_[order[:5],order[-5:]]

  # plot roll pitch yaw for nplot trajectories
  fig,ax = plot_body_rpy_traj(modeldata,dt,idxplot=idxplot)
    
  # plot wing angles
  #plot_wing_angle_trajs(modeldata,idxplot,dosave=True)
  plot_wing_angle_trajs(modeldata,idxplot[[0,]])
  
  add_wingstroke_timing(modeldata,suffix='_ref')
  add_wingstroke_timing(modeldata)

  fig,ax = plot_attackangle_by_stroketype(modeldata,plotall=True)
  print('Goodbye!')