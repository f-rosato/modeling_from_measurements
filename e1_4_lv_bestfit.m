clear all

load data/population_data
U = pop';

%normalize the data to avoid disproportionate Theta
U = U/max(max(U));

% derivatives
Up = diff(U);

% garbage in, garbage out. These derivatives are noisy
sUp = smoothdata(Up);
% sUp = Up;

figure(2)
subplot(2,1,1)
plot(Up(:,1), 'Linewidth', [3]);
hold on
grid on
plot(sUp(:,1), 'Linewidth', [3]);
legend('raw', 'smoothed')
ylabel('norm. population')
xlabel('year')
xticklabels(year)
title('Hare')

subplot(2,1,2)
plot(Up(:,2), 'Linewidth', [3]);
hold on
grid on
plot(sUp(:,2), 'Linewidth', [3]);
legend('raw', 'smoothed')
ylabel('norm. population')
xlabel('year')
xticklabels(year)
title('Linx')

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
names_gr = {['Hare '], ['Linx ']};
figure(1)
sp = 1;
for ix = 1:2
    
    % do lasso
    [B, FitInfo] = lasso(Theta, Up(:,ix), 'DFmax', 4); %  'CV', 3, 'DFmax', 4);

    % select only the good stuff
    nonzero_rows = find(~all(B==0,2));
    names = Theta_names(nonzero_rows);
    B_vip = B(nonzero_rows, :);
    
    lambda_labels = cell(1,length(FitInfo.Lambda));
    for ii = 1:length(FitInfo.Lambda)
        lambda_labels{ii} = sprintf('%0.2e',FitInfo.Lambda(ii));
    end
    subplot(4,1,sp);
    sp = sp + 1;
    heatmap(lambda_labels, names, B_vip, 'Colormap', colormap('jet'), 'XLabel', 'Lambda', 'YLabel', 'function')
    title([names_gr{ix}, '- SINDy modes vs lambda (raw data)'])
    
end



for ix = 1:2
    
    % do lasso
    [B, FitInfo] = lasso(Theta, sUp(:,ix), 'DFmax', 4); %  'CV', 3, 'DFmax', 4);

    % select only the good stuff
    nonzero_rows = find(~all(B==0,2));
    names = Theta_names(nonzero_rows);
    B_vip = B(nonzero_rows, :);
    
    lambda_labels = cell(1,length(FitInfo.Lambda));
    for ii = 1:length(FitInfo.Lambda)
        lambda_labels{ii} = sprintf('%0.2e',FitInfo.Lambda(ii));
    end

    subplot(4,1,sp);
    sp = sp + 1;
    heatmap(lambda_labels, names, B_vip, 'Colormap', colormap('jet'), 'XLabel', 'Lambda', 'YLabel', 'function')
    title([names_gr{ix}, '- SINDy modes vs lambda (data smoothed)'])
    
end