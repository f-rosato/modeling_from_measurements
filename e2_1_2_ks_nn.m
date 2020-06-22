clear all; close all

%% generate the data
init_cond_type = 2;
% hiddenLayerSize = [12,9,3];

[u, t, x] = ks_uu(init_cond_type);
[nts0, nxs0] = size(u);
nts = nts0;
nxs = nxs0;
% figure(1);
% set(gcf,'color','w');
% title('Original KS data')
% pcolor(x,t,u), shading interp, colormap(hot), %view(15,50)

% matrix u
%             x(2048 pts)
% -------------------------->
% |
% |
% |
% |t (59 steps)
% v

%% preparation of the data to feed to the neural network
x_window = 9;
t_window = 2;

data_mtx = zeros([x_window * t_window, 1]);
target_v = 0;

%       prediction mask
% [] [] [] [] [] [] [] [] []   % sample (x_window X time_window reshaped to nx1)
% [] [] [] [] [] [] [] [] []   
%             {}               % target

% each sample is a column of data_mtx
% each target is an element of target_v

cont = 1;
for ts = 1:nts-t_window
    for xs = 1:nxs-x_window + 1
        
        x_central = xs + (x_window - 1) / 2;
        t_central = ts + t_window;
        
        t_slice = ts: ts + t_window - 1;
        x_slice = xs: xs + x_window - 1;

        data = u(t_slice, x_slice);
        data = reshape(data, [1, x_window * t_window,])';
        target = u(t_central, x_central);
        
        data_mtx(:,cont) = data;
        target_v(cont) = target;
        
        cont = cont + 1;
    end
end

%% prepared data is 0-1 normalized
max_v = max(max(data_mtx));
min_v = min(min(data_mtx));

scale_down = @(x) (x - min_v) / (max_v - min_v);
scale_up = @(x) x * (max_v - min_v) +  min_v;

data_mtx = scale_down(data_mtx);
target_v = scale_down(target_v);


%% TRAINING HAPPENS HERE.....
% nn_train_regr  % training script uses data_mtx and target_v; outputs net
% save('ksnet_3.mat', net)
% uncomment the line above for retraining

%% EVALUATION OF PREDICTION
% load my trained network

% IMPORTANT:
% ksnet_2 is a 9x1 net
% ksnet_1 is a 9x2 net
load('saved_nets/ksnet_1.mat')

% we predict and reshape the result and confront it
predicted_v = net(data_mtx);
predicted_u = scale_up(reshape(predicted_v, [length(predicted_v)/(nts0 - t_window), (nts0 - t_window)])');

% since the prediction uses the two initial rows and a few columns 
% on the side, here we shave the indices that were not predicted
shaved_u = u(t_window+1:end, 1 +(x_window - 1)/2: 2048 - (x_window - 1)/2);
shaved_x = x(1 +(x_window - 1)/2 : 2048 - (x_window - 1)/2);
shaved_t = t(t_window+1:end);

figure(2)
set(gcf,'color','w');
subplot(1,2,2)
pcolor(shaved_x,shaved_t,predicted_u), shading interp, colormap(hot), %view(15,50)
title('Each predicted from original data')
xlabel('space (norm.)')
ylabel('time (norm.)')


subplot(1,2,1)
pcolor(shaved_x,shaved_t,shaved_u), shading interp, colormap(hot), %view(15,50)
title(['Original data (init cond:', num2str(init_cond_type), ')'])
xlabel('space (norm.)')
ylabel('time (norm.)')


%% RECURSIVE PREDICTION
% now we try to recursively feed the neural network with its own prediction
recurs_predicted_u = zeros(nts0 - t_window, nxs0);

first_u = u(1:t_window,:);
[nts, nxs] = size(first_u);

rolling_u = scale_down(first_u);

row_cont = 1;

for ttr = t_window + 1:nts0

    line_mtx = zeros([x_window * t_window, 1]);

    cont = 1;
    for ts = 1:nts-t_window + 1
        for xs = 1:nxs-x_window + 1

            t_slice = ts: ts + t_window - 1;
            x_slice = xs: xs + x_window - 1;

            data = rolling_u(t_slice, x_slice);
            data = reshape(data, [1, x_window * t_window,])';

            line_mtx(:,cont) = data;

            cont = cont + 1;
        end
    end
    next_predicted_line = [zeros(1, (x_window - 1)/2), net(line_mtx), zeros(1, (x_window - 1)/2)];

    recurs_predicted_u(row_cont, :) = next_predicted_line;
    row_cont = row_cont + 1;
    rolling_u = next_predicted_line;
    
    for at =1:t_window - 1
        rolling_u = [rolling_u(end + 1 - at,:) ; rolling_u];
    end
end

recurs_predicted_u = scale_up(recurs_predicted_u);

figure(3)
set(gcf,'color','w');
subplot(1,2,2)
pcolor(x,shaved_t,recurs_predicted_u), shading interp, colormap(hot), %view(15,50)
title('Rolling prediction')
xlabel('space (norm.)')
ylabel('time (norm.)')

subplot(1,2,1)
pcolor(x,shaved_t,u(1+t_window:end,:)), shading interp, colormap(hot), %view(15,50)
title(['Original data (init cond:', num2str(init_cond_type), ')'])
xlabel('space (norm.)')
ylabel('time (norm.)')


