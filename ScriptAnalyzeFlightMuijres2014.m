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
  omega_computed(j0:j1-1,k,:) = omegacurr; % rad / frame
  rpy_curr = cumsum([zeros(1,3);omegacurr]);
  rpy_computed(j0:j1,k,:) = rpy_curr;

  % omega is in units of rad/s
  omegacurr = squeeze(body_data.omega(:,k,:))*dt;
  omegacurr = omegacurr(j0:j1-1,:);
  omega_loaded(j0:j1-1,k,:) = omegacurr;
  rpy_curr = cumsum([zeros(1,3);omegacurr]);
  rpy_loaded(j0:j1,k,:) = rpy_curr;

end

figure(1);
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


figure(2);
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

figure(3);
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

k = 75;
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

figure(4);
clf;
hax = createsubplots(4,3,.05);
hax = reshape(hax,[4,3]);
lw = 2;

for i = 1:3,
  plot(hax(i,1),pos(:,i),'-','LineWidth',lw);
  title(hax(i,1),sprintf('pos%d',i));
  plot(hax(i,2),vel0(:,i),'-','LineWidth',lw);
  hold(hax(i,2),'on');
  plot(hax(i,2),vel1(:,i),'--','LineWidth',lw);
%   ylim = get(hax(i,2),'ylim');
%   set(hax(i,2),'ylim',ylim);
  title(hax(i,2),sprintf('vel%d',i));
  plot(hax(i,3),acc0(:,i),'-','LineWidth',lw);
  hold(hax(i,3),'on');
  plot(hax(i,3),acc1(:,i),'--','LineWidth',lw);
%   ylim = get(hax(i,3),'ylim');
%   set(hax(i,3),'ylim',ylim);
  title(hax(i,3),sprintf('acc%d',i));
end
plot(hax(4,3),accmag_g,'LineWidth',lw);
hold(hax(4,3),'on');
plot(hax(4,3),[1,j1-j0+1],acc_thresh_g+[0,0],'k-');
title(hax(4,3),'accmag (g)');

legend(hax(1,2),{'loaded','computed'});

%% compute response times

ntraj = size(body_data.pos,2);
nc = ceil(sqrt(ntraj));
nr = ceil(ntraj/nc);
figure(5);
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
figure(6);
clf;
plot(repmat(1:ntraj,[2,1]), [startts,stimstart+zeros(ntraj,1)]','b-');
hold on;
plot(repmat(1:ntraj,[2,1]), [stimstart+zeros(ntraj,1),responsets]','g-');
plot(1:ntraj,startts','.');
plot(1:ntraj,stimstart+zeros(1,ntraj),'.');
plot(1:ntraj,responsets,'.');

%% starts and ends of wing strokes
thresh_factor = .5;

upstrokes = cell(1,ntraj);
downstrokes = cell(1,ntraj);
strokestarts = cell(1,ntraj);
strokeends = cell(1,ntraj);
linear_acc_stroke = cell(1,ntraj);
angular_acc_stroke = cell(1,ntraj);
for i = 1:ntraj,
  js = find(~isnan(wing_data.stroke_L(:,i)));
  j0_wing = js(1);
  j1_wing = js(end);
  % wing and body data seem to be offset from each other
  % use relative frame
  upstrokes{i} = find(islocalmin(wing_data.stroke_L(j0_wing:j1_wing,i)));
  downstrokes{i} = find(islocalmax(wing_data.stroke_L(j0_wing:j1_wing,i)));
  % stroke period goes from downstroke start to upstroke end
  strokestarts{i} = downstrokes{i}(1:end-1);
  strokeends{i} = downstrokes{i}(2:end)-1;
end

mean_stroke_dur = mean(cat(1,strokeends{:})-cat(1,strokestarts{:}))+1;
std_stroke_dur = std(cat(1,strokeends{:})-cat(1,strokestarts{:}));

for i = 1:ntraj,
  j0_body = find(~isnan(body_data.accel(:,i,1)),1);
  linear_acc_stroke{i} = nan(numel(strokestarts{i}),1);
  angular_acc_stroke{i} = nan(numel(strokestarts{i}),3);
  for j = 1:numel(strokestarts{i}),
    t0 = strokestarts{i}(j)+j0_body-1;
    t1 = strokeends{i}(j)+j0_body-1;

    % acc in m / s^2
    linear_acccurr = body_data.accel(t0:t1,i,:);
    assert(~any(isnan(linear_acccurr(:))));
    meanacc = mean(linear_acccurr,1);
    meanacc(end) = meanacc(end) + g;
    % unitless
    meanaccmag = vecnorm(meanacc,2)/g;
    linear_acc_stroke{i}(j) = meanaccmag;

    % omega is in units of rad/s -> deg / stroke
    omegacurr = squeeze(body_data.omega(t0:t1,i,:))*180/pi*dt*mean_stroke_dur;
    % omegadot is deg / (wing beat)^2
    omegadotcurr = (omegacurr(end,:)-omegacurr(1,:));
    angular_acc_stroke{i}(j,:) = abs(omegadotcurr);

  end
end

% this is as close as i could get to the thresholds in the paper...
mean_linear_acc_stroke = mean(cat(1,linear_acc_stroke{:}));
std_linear_acc_stroke = std(cat(1,linear_acc_stroke{:}),1);
thresh_linear_acc_stroke = 1 + thresh_factor*std_linear_acc_stroke;

mean_angular_acc_stroke = mean(cat(1,angular_acc_stroke{:}),1);
std_angular_acc_stroke = std(cat(1,angular_acc_stroke{:}),1,1);
thresh_angular_acc_stroke = thresh_factor*std_angular_acc_stroke;

nstrokes_total = sum(cellfun(@numel,strokestarts));

issteady_stroke = cell(ntraj,1);
nsteady_per = nan(ntraj,4);
nsteady = nan(ntraj,1);
nstrokes = nan(ntraj,1);
for i = 1:ntraj,
  nsteady_per(i,:) = sum(cat(2,linear_acc_stroke{i},angular_acc_stroke{i}) <= ...
    cat(2,thresh_linear_acc_stroke,thresh_angular_acc_stroke),1);
  issteady_stroke{i} = (linear_acc_stroke{i} <= thresh_linear_acc_stroke) & ...
    all(angular_acc_stroke{i} <= thresh_angular_acc_stroke,2);
  nsteady(i) = nnz(issteady_stroke{i});
  nstrokes(i) = numel(strokestarts{i});
end

%% plot acceleration per wing stroke
i = 1;
figure(7);
hax = createsubplots(4,1,.05);
t0_body = find(~isnan(body_data.accel(:,i,1)),1);
colors = flipud(lines(2));
coords = {'x','y','z'};
for j = 1:4,
  if j == 4,
    title(hax(j),'acc mag');
  else
    title(hax(j),sprintf('|d omega_%s|',coords{j}));
  end
  hold(hax(j),'on');
end
for j = 1:4,
  if j == 4,
    datacurr = squeeze(body_data.accel(:,i,:));
    datacurr(:,end) = datacurr(:,end)+g;
    datacurr = vecnorm(datacurr,2,2)/g;
    strokedatacurr = linear_acc_stroke{i};
  else
    datacurr = body_data.omega(:,i,j)*180/pi*dt*mean_stroke_dur;
    datacurr = abs(diff(datacurr))*mean_stroke_dur;
    strokedatacurr = angular_acc_stroke{i}(:,j);
  end
  for k = 1:numel(strokestarts{i}),
    t0 = strokestarts{i}(k)+t0_body-1;
    t1 = strokeends{i}(k)+t0_body-1;
    color = colors(double(issteady_stroke{i}(k))+1,:);
    plot(hax(j),t0:t1,datacurr(t0:t1),'k-');
    plot(hax(j),[t0,t1],[0,0]+strokedatacurr(k),'.-','color',color);
  end
end

hunsteady = plot(hax(1),nan,nan,'.-','color',colors(1,:));
hsteady = plot(hax(1),nan,nan,'.-','color',colors(2,:));
legend([hunsteady,hsteady],{'Unsteady','Steady'});

%%

figure(8);
clf;
image(cat(3,~isnan(body_data.accel(:,:,1)),zeros(size(wing_data.stroke_L)),~isnan(wing_data.stroke_L)));

%%

[qo,qv] = prepRotate(q,v,'rotatepoint');
q = [0.707, 0.0,  0.707, 0.0];
v = [1,0,0];
qo = q(:);
qv = [0,v];


qo = reshape(q,[  ],1);
qo = normalize(qo);
%Make v a quaternion
z = zeros(nv,1,'like',v);
qv = quaternion([ z,v ]);


uq = qo .* qv .* conj(qo);
up = compact(uq);
u = up(:,2:end);
