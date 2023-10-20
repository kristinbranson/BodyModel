load FRUITFLY_LOOMINGRESPONSE_DATABASE.mat;
addpath ~/behavioranalysis/code/Jdetect/Jdetect/misc;

%% compute roll pitch yaw from quaternion body position

ntraj = size(body_data.pos,2);
T = size(body_data.pos,1);
dt = 1/settings_variables.fps;

rpy_loaded = nan([T,ntraj,3]);
rpy_computed = nan([T,ntraj,3]);
omega_loaded = nan([T,ntraj,3]);
omega_computed = nan([T,ntraj,3]);
dts = (0:T-1)*dt;
for k = 1:ntraj,
  q = squeeze(body_data.qbody(:,k,:));
  q = q(:,[4,1,2,3]);

  js = find(~isnan(q(:,1)));
  j0 = js(1);
  j1 = js(end);
  dq = nan(j1-j0,4);
  for j = j0:j1-1,
    dq(j-j0+1,:) = quatmultiply(quatconj(q(j,:)),q(j+1,:));
  end

  % scalar is first term
  droll  = atan2(2.0 * (dq(:,4) .* dq(:,3) + dq(:,1) .* dq(:,2)) , 1.0 - 2.0 * (dq(:,2) .* dq(:,2) + dq(:,3) .* dq(:,3)));
  dpitch = asin(2.0 * (dq(:,3) .* dq(:,1) - dq(:,4) .* dq(:,2)));
  dyaw   = atan2(2.0 * (dq(:,4) .* dq(:,1) + dq(:,2) .* dq(:,3)) , - 1.0 + 2.0 * (dq(:,1) .* dq(:,1) + dq(:,2) .* dq(:,2)));

%   % scalar is last term
%   droll  = atan2(2.0 .* (dq(:,3) .* dq(:,2) + dq(:,4) .* dq(:,1)) , 1.0 - 2.0 .* (dq(:,1) .* dq(:,1) + dq(:,2) .* dq(:,2)));
%   dpitch = asin(2.0 .* (dq(:,2) .* dq(:,4) - dq(:,3) .* dq(:,1)));
%   dyaw   = atan2(2.0 .* (dq(:,3) .* dq(:,4) + dq(:,1) .* dq(:,2)) , - 1.0 + 2.0 .* (dq(:,4) .* dq(:,4) + dq(:,1) .* dq(:,1)));
  omegacurr = [droll,dpitch,dyaw];
  omega_computed(j0:j1-1,k,:) = omegacurr;
  rpy_curr = cumsum([zeros(1,3);omegacurr]);
  rpy_computed(j0:j1,k,:) = rpy_curr;

  omegacurr = squeeze(body_data.omega(:,k,:))*dt;
  omegacurr = omegacurr(j0:j1-1,:);
  omega_loaded(j0:j1-1,k,:) = omegacurr;
  rpy_curr = cumsum([zeros(1,3);omegacurr]);
  rpy_loaded(j0:j1,k,:) = rpy_curr;

end

figure(6);
clf;
hax = createsubplots(3,10);
hax = reshape(hax,[3,10]);

for k = 1:10,
  rpy_curr = squeeze(rpy_computed(:,k,:));
  js = find(~isnan(rpy_curr(:,1)));
  j0 = js(1);
  j1 = js(end);
  rpy_curr = rpy_curr(j0:j1,:);
  for i = 1:3,
    plot(hax(i,k), rpy_curr(:,i),'b.-');
  end

  rpy_curr = squeeze(rpy_loaded(j0:j1,k,:));
  for i = 1:3,
    hold(hax(i,k),'on');
    plot(hax(i,k), rpy_curr(:,i),'r.-');
  end
end


figure(7);
clf;
hax = createsubplots(3,10);
hax = reshape(hax,[3,10]);

for k = 1:10,
  omega_curr = squeeze(omega_computed(:,k,:));
  js = find(~isnan(omega_curr(:,1)));
  j0 = js(1);
  j1 = js(end);
  omega_curr = omega_curr(j0:j1,:);
  for i = 1:3,
    plot(hax(i,k), omega_curr(:,i),'b.-');
  end

  omega_curr = squeeze(omega_loaded(j0:j1,k,:));
  for i = 1:3,
    hold(hax(i,k),'on');
    plot(hax(i,k), omega_curr(:,i),'r.-');
  end
end
legend({'computed','loaded'});
ylabel(hax(end,1),'omega')

%save rpy.mat rpy_computed rpy_loaded dts

%%

traji = 1;
qfloat = squeeze(body_data.qbody(:,traji,:));
qfloat = qfloat(:,[4,1,2,3]);
js = find(~any(isnan(qfloat),2));
j0 = js(1);
j1 = js(end);
T = j1-j0+1;
dts = ones(T,1);
qfloat = qfloat(j0:j1,:);
q = quaternion(qfloat);
qi = quaternion.ones(1,1);

q = normalize(q);

qdiff_active = q(2:end) .* conj(q(1:end-1));
qdiff_passive = conj(q(1:end-1)) .* q(2:end);

qfloat_diff_passive = quatmultiply(quatconj(qfloat(1:end-1,:)),qfloat(2:end,:));

av_passive = compact((2./1) .* qdiff_passive);
av_passive = av_passive(:,2:4);

av_float_passive = 2*qfloat_diff_passive(:,2:end);

dq = qfloat_diff_passive;
% scalar is first term
droll  = atan2(2.0 * (dq(:,4) .* dq(:,3) + dq(:,1) .* dq(:,2)) , 1.0 - 2.0 * (dq(:,2) .* dq(:,2) + dq(:,3) .* dq(:,3)));
dpitch = asin(2.0 * (dq(:,3) .* dq(:,1) - dq(:,4) .* dq(:,2)));
dyaw   = atan2(2.0 * (dq(:,4) .* dq(:,1) + dq(:,2) .* dq(:,3)) , - 1.0 + 2.0 * (dq(:,1) .* dq(:,1) + dq(:,2) .* dq(:,2)));
omegacurr = [droll,dpitch,dyaw];

figure(234);
clf;
hax = createsubplots(3,1,.05);
for i = 1:3,
  plot(hax(i),av_passive(:,i),'o-');
  hold(hax(i),'on');
  plot(hax(i),av_float_passive(:,i),'x-');
  plot(hax(i),omegacurr(:,i),'s-');
  plot(hax(i),omega_computed(j0:j1,traji,i),'^-');
  plot(hax(i),omega_loaded(j0:j1,traji,i),'.-');
end
legend({'quaternion passive','float passive','omegacurr','omega computed','omega loaded'});

%% compare computed and loaded pos, vel, acc

g = 9.80665; % m / (s^2)
t0 = settings_variables.trigger_frame;
acc_thresh_g = 0.28;
minresponsei = 50;

k = 27;
pos = squeeze(body_data.pos(:,k,:));
js = find(~isnan(pos(:,1)));
j0 = js(1);
j1 = js(end);
pos = pos(j0:j1,:);
vel0 = squeeze(body_data.vel(j0:j1,k,:));
vel1 = (pos(2:end,:)-pos(1:end-1,:))/dt;
acc0 = squeeze(body_data.accel(j0:j1,k,:));
acc1 = (vel0(2:end,:)-vel0(1:end-1,:))/dt;
accmag = sqrt(sum(acc0.^2,2));
accmag_g = accmag / g;

figure(1);
clf;
hax = createsubplots(4,2,.05);
hax = reshape(hax,[4,2]);
lw = 2;

for i = 1:3,
  plot(hax(i,1),vel0(:,i),'-','LineWidth',lw);
  hold(hax(i,1),'on');
  plot(hax(i,1),vel1(:,i),'--','LineWidth',lw);
  ylim = get(hax(i,1),'ylim');
  set(hax(i,1),'ylim',ylim);
  title(hax(i,1),sprintf('vel%d',i));
  plot(hax(i,2),acc0(:,i),'-','LineWidth',lw);
  hold(hax(i,2),'on');
  plot(hax(i,2),acc1(:,i),'--','LineWidth',lw);
  ylim = get(hax(i,2),'ylim');
  set(hax(i,2),'ylim',ylim);
  title(hax(i,2),sprintf('acc%d',i));
end
axes(hax(1,1));
legend({'loaded','computed'});
plot(hax(4,2),accmag_g,'LineWidth',lw);
hold(hax(4,2),'on');
plot(hax(4,2),[1,j1-j0+1],acc_thresh_g+[0,0],'k-');
title(hax(4,2),'accmag (g)');

legend(hax(1,1),{'loaded','computed'});

%% compute response times

ntraj = size(body_data.pos,2);
nc = ceil(sqrt(ntraj));
nr = ceil(ntraj/nc);
figure(2);
clf;
hax = createsubplots(nr,nc);
hax = hax(:);
startts = nan(ntraj,1);
responsets = nan(ntraj,1);
responsets_local = nan(ntraj,1);
stimstart = (t0-1)*dt;
for k = 1:ntraj,
  pos = squeeze(body_data.pos(:,k,:));
  js = find(~isnan(pos(:,1)));
  j0 = js(1);
  j1 = js(end);
  startts(k) = (j0-1)*dt;
  acc0 = squeeze(body_data.accel(j0:j1,k,:));
  accmag = sqrt(sum(acc0.^2,2));
  accmag_g = accmag / g;
  i0 = max(1,t0 - j0 + 1);
  % find where the acc goes below threshold
  i1 = find(accmag_g(i0:end)<=acc_thresh_g,1)+i0-1;
  i0 = i1;
  [maxacc,i2] = max(accmag_g(i0:end));
  i2 = i2+i0-1;
  i1 = find(accmag_g(i0:i2)<=acc_thresh_g,1,'last')+i0-1;
  %i1 = find(accmag_g(i0:end)>=acc_thresh_g,1)+i0-1;
  responsets(k) = (i1+j0-1-1)*dt;
  responsets_local(k) = (i1-1)*dt;
  plot(hax(k),(j0-1:j0-1+numel(accmag_g)-1)*dt,accmag_g);
  axisalmosttight([],hax(k));
  hold(hax(k),'on');
  ylim = get(hax(k),'ylim');
  plot(hax(k),[stimstart,stimstart],ylim,'g-');
  plot(hax(k),[responsets(k),responsets(k)],ylim,'k-');
  axisalmosttight([],hax(k));
  set(hax(k),'ylim',ylim);
  title(hax(k),num2str(k));
end

%% plot response times
figure(3);
clf;
plot(repmat(1:ntraj,[2,1]), [startts,stimstart+zeros(ntraj,1)]','b-');
hold on;
plot(repmat(1:ntraj,[2,1]), [stimstart+zeros(ntraj,1),responsets]','g-');
plot(1:ntraj,startts','.');
plot(1:ntraj,stimstart+zeros(1,ntraj),'.');
plot(1:ntraj,responsets,'.');
