import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm,animation,colors
from matplotlib.patches import Rectangle
import pickle
import scipy.ndimage as ndimage
import copy

anglenames = ['yaw','roll','pitch']
bodycomnames = ['x','y','z']
bodyquatnames = ['theta','x','y','z']
angleidx = {k: i for i,k in enumerate(anglenames)}
bodycomidx = {k: i for i,k in enumerate(bodycomnames)}
bodyquatidx = {k: i for i,k in enumerate(bodyquatnames)}

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

  body_pitch_angle = np.deg2rad(body_pitch_angle)
  # Yaw: doesn't require transformation.
  # Roll.
  for i in range(angles.shape[-1]//3):
    off = i*3
    # yaw doesn't require transformation
    
    # roll
    angles[:, off+angleidx['roll']] = - angles[:, off+angleidx['roll']]

    # pitch
    angles[:, off+angleidx['pitch']] = np.pi/2 - body_pitch_angle - angles[:, off+angleidx['pitch']]

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
    plt.plot(x_axis, angles[:, 3], color_right, linewidth=linewidth,label='Right')
    plt.plot(x_axis, angles[:, 0], color_left, linewidth=linewidth,label='Left')
    plt.legend()
    # Downstrokes.
    ylim = plt.ylim()
    for maximum, minimum in zip(maxima, minima):
        ax.add_patch(Rectangle((factor*dt*maximum, -100), factor*dt*(minimum-maximum), 200, color='k', alpha=downstroke_alpha))
    plt.ylim(ylim)
    plt.ylabel('Yaw\n(deg)', fontsize=fs_labels)
    plt.yticks(fontsize=fs_ticks)
    # == Roll.
    ax = plt.subplot(3, 1, 2)
    axs.append(ax)
    plt.plot(x_axis, angles[:, 4], color_right, linewidth=linewidth,label='Right')
    plt.plot(x_axis, angles[:, 1], color_left, linewidth=linewidth,label='Left')
    ylim = plt.ylim()
    # Downstrokes.
    for maximum, minimum in zip(maxima, minima):
        ax.add_patch(Rectangle((factor*dt*maximum, -100), factor*dt*(minimum-maximum), 200, color='k', alpha=downstroke_alpha))
    plt.ylim(ylim)
    plt.ylabel('Roll\n(deg)', fontsize=fs_labels)
    plt.yticks(fontsize=fs_ticks)
    # == Pitch.
    ax = plt.subplot(3, 1, 3)
    axs.append(ax)
    plt.plot(x_axis, angles[:, 5], color_right, linewidth=linewidth,label='Right')
    plt.plot(x_axis, angles[:, 2], color_left, linewidth=linewidth,label='Left')
    ylim = plt.ylim()
    # Downstrokes.
    for maximum, minimum in zip(maxima, minima):
        ax.add_patch(Rectangle((factor*dt*maximum, -100), factor*dt*(minimum-maximum), 200, color='k', alpha=downstroke_alpha))
    plt.ylim(ylim)
    plt.ylabel('Pitch\n(deg)', fontsize=fs_labels)
    plt.xlabel('Time (ms)', fontsize=fs_labels)
    plt.xticks(fontsize=fs_ticks)
    plt.yticks(fontsize=fs_ticks)
    
    return fig,np.array(axs)

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
      print(f'filterint list {k}')
      data[k] = [v[i] for i in trajidx]
    elif (type(v) == np.ndarray) and v.shape[0] == ntraj:
      print(f'filtering ndarray {k}')
      data[k] = v[trajidx]
    else:
      print(f'skipping {k}')
  return

if __name__ == "__main__":
  print('Hello!')

  accmag_thresh = .28 * 9.81 * 1e2 # 0.28 g  
  #testtrajidx = np.array([  5,   6,   8,  13,  14,  19,  22,  23,  24,  32,  43,  57,  59, 68,  75,  79,  85,  87,  90, 101, 103, 114, 117, 118, 120, 131, 132, 134, 137, 138, 148, 150, 153, 157, 173, 183, 185, 190, 194, 204, 205, 211, 213, 214, 216, 223, 226, 233, 241, 246, 258, 266, 268, 270, 271])
  testtrajidx = np.array([  5,   6,   8,  13,  14,  19,  22,  23,  24,  32,  43,  57,  59, 68,  75,  79,  85,  87,  90, 101, 103, 114, 117, 118, 120, 131, 132, 134, 137, 138, 148, 150, 153, 157, 173, 183, 185, 190, 194, 204, 205, 211, 213, 214, 216, 223, 226, 233, 241, 246, 258, 266, 268, 270, 271])
  realdatafile = 'flight-dataset_wing-qpos_not-augmented_evasion-saccade_n-traj-136.pkl'
  modeldatafile = 'analysis-rollouts-272_flight-imitation-wbpg-shaping_no-root-quat-no-height-obs_init-rot-qvel_data-hdf5_start-step-random_net-1_wing-prms-18_split-train-test_margin-0.4-pi_seed-1.pkl'
  realdata = load_data(realdatafile)
  allmodeldata = load_data(modeldatafile)
  
  modeldata = copy.deepcopy(allmodeldata)
  filter_trajectories(modeldata,testtrajidx)
    
  dt = realdata['dt']
  deltaturn = int(40e-3/dt) # frame

  # compute roll pitch yaw
  add_rpy_to_data(modeldata)
  add_rpy_to_data(modeldata,suffix='_ref')
  
  # compute vel and acc
  add_vel_acc_to_data(modeldata,dt)
  add_vel_acc_to_data(modeldata,dt,suffix='_ref')
  
  # compute response times
  add_response_times(modeldata,accmag_thresh,suffix='_ref')

  # compute turn angles
  add_turnangles(modeldata,suffix='_ref',deltaturn=deltaturn)
  
  fig,ax = plot_rpy_response(modeldata,dt,deltaturn=deltaturn)
  
  # remove the repeated trajectories from the model data
  # idxkeep = np.r_[np.arange(92),np.arange(184,228)]
  # ntraj0 = len(modeldata['qpos'])
  # istesttraj = np.zeros(len(modeldata['qpos']),dtype=bool)
  # istesttraj[testtrajidx] = True
  
  # for k,v in modeldata.items():
  #   if type(v) == list and len(v) == ntraj0:
  #     print(f'filtering {k}')
  #     modeldata[k] = [v[i] for i in idxkeep]
  #   else:
  #     print(f'skipping {k}')
  # istesttraj = istesttraj[idxkeep]
  # testtrajidx = np.nonzero(istesttraj)[0]
  # ntraj_model = len(modeldata['qpos'])
  # ntraj_real = len(realdata['qpos'])
  # assert ntraj_model == ntraj_real
  # ntraj = ntraj_model
  ntraj = len(modeldata['qpos'])
  
  order = np.argsort(modeldata['turnangle'][testtrajidx])
  order = testtrajidx[order]
  idxplot = np.r_[order[:5],order[-5:]]

  # plot roll pitch yaw for first nplot trajectories
  fig,ax = plot_body_rpy_traj(modeldata,dt,idxplot=idxplot)

  
  # plot wing angles
  traji = testtrajidx[1]
  figreal,axreal = plot_wing_angles(realdata['wing_qpos'][traji],transform_model_to_data=True)
  figmodel,axmodel = plot_wing_angles(modeldata['wing_qpos'][traji],transform_model_to_data=True)
  axmodel[0].set_title('Model')
  axreal[0].set_title('Real')
  # link axes for the real and model plots
  for i in range(len(axreal)):
    ylimreal = axreal[i].get_ylim()
    ylimmodel = axmodel[i].get_ylim()
    ylim = (min(ylimreal[0],ylimmodel[0]),max(ylimreal[1],ylimmodel[1]))
    axreal[i].set_ylim(ylim)
    axmodel[i].set_ylim(ylim)


  
  print('Goodbye!')