clear all

load data/population_data
U = pop';

%normalize the data to avoid disproportionate Theta
U = U/max(max(U));

% derivatives
Up = diff(U);

% garbage in, garbage out. These derivatives are noisy
sUp = smoothdata(Up);

% plot(xp);
% hold on
% plot(sxp);

% shave the last data point with no known fwd derivative
U = U(1:end-1, :); 


% construct a library-matrix of possible dynamics terms
x = U(:,1);
y = U(:,2);

Theta = zeros(size(U,1), 10);
Theta_names = {'1', 'x', 'y', 'xy', 'xx', 'yy', 'xxy', 'yyx', 'sinx', 'cosx'};

Theta(:,1) = ones(size(x));
Theta(:,2) = x;
Theta(:,3) = y;
Theta(:,4) = x.*y;
Theta(:,5) = x.^2;
Theta(:,6) = y.^2;
Theta(:,7) = y.*(x.^2);
Theta(:,8) = x.*(y.^2);
Theta(:,9) = sin(x);
Theta(:,10) = sin(y);

Csi = zeros(10, size(U,2));

for ix = 1:2
    
    % do lasso
    [B, FitInfo] = lasso(Theta, sUp(:,ix), 'DFmax', 4); %  'CV', 3, 'DFmax', 4);

    % select only the good stuff
    nonzero_rows = find(~all(B==0,2));
    names = Theta_names(nonzero_rows);
    B_vip = B(nonzero_rows, :);
    
    % in order to evaluate the results, we create a custom
    % scatter-stem plot of the selected coefficients varying with lambda
    figure
    scatter_x = 0;
    scatter_y = 0;
    scatter_z = 0;
    
    [ll, cc] = meshgrid(FitInfo.Lambda, 1:length(nonzero_rows));
    index = 1;
    for r = 1:size(ll,1)
        for c = 1:size(ll,2)
            if B_vip(r,c) ~= 0
                scatter_x(index) = ll(r,c);
                scatter_y(index) = cc(r,c);
                scatter_z(index) = B_vip(r,c);
                index = index + 1;
            end
        end
    end
        
    stem3(scatter_x, scatter_y, scatter_z+1, ':d', 'LineWidth', 1);
    yticks(1:length(nonzero_rows));
    yticklabels(names);
    zlabel('coefficient')
    xlabel('lambda')
    % lassoPlot(B,FitInfo,'PlotType','CV');
end

% clear all
% X = randn(100,5);
% weights = [0;2;0;-3;0]; % Only two nonzero coefficients
% y = X*weights + randn(100,1)*0.1; % Small added noise
% B = lasso(X,y);