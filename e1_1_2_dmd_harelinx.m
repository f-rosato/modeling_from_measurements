clear all; close all; clc

% loads data
load data/population_data.mat
v = pop;

%% settings ---------------------------------------------------
% number of steps back in the time delay vector 
n = 3;
% rank of the approximation 
r = 3;
% how many "test" results to keep aside
n_test = 7;

%notes: --------------------------------------------------------
% r=0 uses all SVD modes; it's the same as r = n*2
% n=1, r=2 we use just the data with no time delay embedding
%---------------------------------------------------------------


%% data manipulation
% computes full time-delay matrix
X_tot = vectd(v(1,:), v(2,:), size(v,2) - n, n);

% computes X and X'
X = X_tot(:,1:end - 1 -n_test);
Xp = X_tot(:,2:end - n_test);

% matrix of test cases
X_test = X_tot(:,end - n_test + 1 :end);

%% core calculations
if r==0
    A = Xp*pinv(X);
else
    [U,Sigma,V] = svd(X, 'econ');
    U_r = U(:,1:r);
    Sigma_r = Sigma(1:r,1:r);
    V_r = V(:,1:r);
    A_tilde = U_r'*Xp*V_r/Sigma_r;
    
    % if we chose to do the reduced rank dmd, we assess with a plot
    % the amplitudes of the singular values
    figure(2)
    plotsigma(Sigma,r)
end

%% predictions
% compute last "known" state
x_l = Xp(:,end);
if r > 0
    x_l_tilde = U_r\x_l;  % POD coefficients vector
end

% advance the state with the linear system
X_predicted = zeros(n*2, n_test);
for t = 1:n_test
    if r==0
        X_predicted(:, t) = A^t*x_l;
    else
        
        % ALTERNATIVE CALCULATION MODE
        % inspired by observation on the book: A_tilde describes the
        % dynamics of the POD coefficients
        x_tilde_predicted = A_tilde^t*x_l_tilde;  % predict POD
        x_predicted = U_r*x_tilde_predicted; % maps back to full state
        
        % CLASSIC CALCULATION MODE
        %x_predicted = U_r * A_tilde * U_r' * x_l;
        %x_l = x_predicted;
        
        X_predicted(:, t) = x_predicted;
    end
end

% since we predicted the whole time-delay vector, only the last two rows
% represent the current situation
X2_predicted = X_predicted(end-1:end,:);
X2_test = X_test(end-1:end,:);
years_predicted = year(end-n_test+1:end);

% let's plot predictions vs harsh reality
if r > 0
    pr_leg = ['(trunc. r=', num2str(r), ')'];
else
    pr_leg = '(All modes)';
end
names = {['Hare ', pr_leg], ['Linx ', pr_leg]};
for index = 1:2
    figure(1)
    subplot(2,1,index)
    plot(years_predicted, X2_test(index,:), 'k', 'Linewidth', [3]);
    hold on
    plot(years_predicted, X2_predicted(index,:), 'r--', 'Linewidth', [3]);
    grid on
    title(names{index})
    ylabel('population')
    xlabel('year')
    xticks(years_predicted)
    legend('actual', 'predicted', 'Location', 'northwest')
end




