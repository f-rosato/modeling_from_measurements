clear all; close all;
load reaction_diffusion_big

%% Preparation of data
% choose measurement and rank
r = 3;
measurement = v;

% flatten the "measurement" from matrix to vector
flat_u = zeros(0);
n = size(measurement,1);
for frame = 1:length(t)
    ur = reshape(measurement(:,:,frame), [1, n^2]);
    flat_u = [flat_u; ur];
    
end
flat_u = flat_u';

%% truncated SVD and singular values plot
[fU, fS, fV] = svd(flat_u, 'econ');

%truncate
tr_fU = fU(:,1:r); tr_fS = fS(1:r,1:r); tr_fV = fV(:,1:r);

% check out the low-rank modes strength
figure(1)
plotsigmalog(fS, r)

%% Plot of the modes
modes = cell(1,r);
figure(2)
for mode_no = 1:r
    fur = tr_fU(:,mode_no);
    flat_v = reshape(fur, [n, n]);
    modes{mode_no} = flat_v;
    
    subplot(1,r, mode_no)
    contourf(x,y,modes{mode_no}); shading interp; colormap(hot)
    title(['Mode #', num2str(mode_no)]);
    colorbar();
end

%% Analysis in the low-rank variables realm
% compute low rank variables
reduced_states = zeros(length(t), r);
legend_items = cell(1,r);
for frame = 1:length(t)
    state = flat_u(:,frame);
    reduced_state = tr_fU\state;
    reduced_states(frame,:) = reduced_state;

end

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

%% Make a NN autoencoder
data_mtx = reduced_states(1:end-1,:)';
target_v = reduced_states(2:end,:)';
hiddenLayerSize = [6,3];
nn_train_regr


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
