clear all; close all;

load data/BZ.mat


% remove the mean from the data
mask = mean(BZ_tensor,3);
figure(1)
pcolor(mask), shading interp
title('Mean of the data (to remove)')
mmask = repmat(mask,1,1,size(BZ_tensor,3));
BZ_tensor = BZ_tensor - mmask;


r = 10;
measurement = BZ_tensor;
n = size(measurement,1);
m = size(measurement,2);

x = 1:m;
y = 1:n;
t = 1:size(BZ_tensor,3);

% flatten the "measurement" from matrix to vector
flat_u = zeros(length(t), n*m);
wb = waitbar(0,'Please wait...');

for frame = 1:length(t)
    waitbar(frame/length(t),wb, 'flattening...')
    ur = reshape(measurement(:,:,frame), [1, n*m]);
    flat_u(frame,:) = ur;
end
close(wb)
flat_u = flat_u';

%% truncated SVD and singular values plot
[fU, fS, fV] = svd(flat_u, 'econ');

%truncate
tr_fU = fU(:,1:r); tr_fS = fS(1:r,1:r); tr_fV = fV(:,1:r);

% check out the low-rank modes strength
figure(1)
plotsigma(fS, r)

%% Plot of the modes
modes = cell(1,r);
figure(2)
for mode_no = 1:r
    fur = tr_fU(:,mode_no);
    flat_v = reshape(fur, [n, m]);
    modes{mode_no} = flat_v;
    
    subplot(round(r/ceil(r/2)),ceil(r/2), mode_no)
    pcolor(x,y,modes{mode_no}); shading interp; colormap(hot)
    title(['Mode #', num2str(mode_no)]);
end

%% Analysis in the low-rank variables realm
% compute low rank variables
reduced_states = zeros(length(t), r);
legend_items = cell(1,r);
wb = waitbar(0,'Please wait...');
for frame = 1:length(t)
    waitbar(frame/length(t),wb, 'computing low-rank...')
    state = flat_u(:,frame);
    reduced_state = tr_fU\state;
    reduced_states(frame,:) = reduced_state;

end
close(wb)

% plot low rank variables
figure(3)
for mode_no = 1:r
    plot(reduced_states(:,mode_no), 'LineWidth', [2])
    hold on
    legend_items{mode_no} = ['low-rank var #', num2str(mode_no)];
end
title('Low-rank variables')
legend(legend_items)
xlabel('frame')

% plot frequencies
figure(4)
Fs = 1;  % 1 frame, by definition
L = length(t);
for mode_no = 1:r
    Y = fft(reduced_states(:,mode_no));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    stem(f(1:30),P1(1:30)) 
    hold on
end
title('Single-Sided Amplitude Spectrum of mode')
xlabel('f (frames^{-1})')
ylabel('|P1(f)|')

% %% Make a NN autoencoder
% data_mtx = reduced_states(1:end-1,:)';
% target_v = reduced_states(2:end,:)';
% hiddenLayerSize = [6,3];
% nn_train_regr


% %% rebuild the approximated 3d matrix
% approx_flat_u = tr_fU * tr_fS * tr_fV';
% approx_u = zeros(size(u));
% for frame = 1:length(t)
%     fur = approx_flat_u(frame,:);
%     flat_u = reshape(fur, [n, n]);
%     approx_u(:,:,frame) = flat_u;
% end
% 
% % compare frames
% nframe = 85;
% 
% figure(3)
% subplot(1,2,1)
% pcolor(x,y,u(:,:,nframe)); shading interp; colormap(hot)
% title(['Original data - frame #', num2str(nframe)])
% subplot(1,2,2)
% pcolor(x,y,approx_u(:,:,nframe)); shading interp; colormap(hot)
% title(['Approximation - r=', num2str(r)])
