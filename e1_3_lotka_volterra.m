clear all

% requires script "population_data.m"
% requires function "lv.m"

population_data;

global year_v
global years_span
global u0
global X_true
global Y_true

year_v = year;
years_span = year(end) - year(1);
u0 = pop(:,1);
X_true = pop(1,:)';
Y_true = pop(2,:)';

% fit optimal parameters
parameters_0 = [37 1.8 0.8 40 25 40];
opt_pars = parameters_0;

options = optimoptions(@fminunc,'Display', 'iter');
[opt_pars, residual] = fminunc(@lotv_dist, parameters_0, options);

disp(residual)
disp('OPTIMAL PARAMETERS:')
disp({'b', 'p', 'r', 'd'})
disp(opt_pars')

lv_stepper_fitted = @(t,u) lv(opt_pars(1), ...
                              opt_pars(2), ...
                              opt_pars(3), ...
                              opt_pars(4), ...
                              t, u);

[t_final, u_final] = ode45(lv_stepper_fitted, [0 1], [opt_pars(5); opt_pars(6)]);

X_model_f = u_final(:,1);
Y_model_f = u_final(:,2);
year_f = year_v(1) + t_final * years_span;

figure(1)
subplot(2,1,1)
plot(year, X_true, 'k', 'Linewidth', [2])
hold on
plot(year_f, X_model_f, 'r--', 'Linewidth', [2])

subplot(2,1,2)
plot(year, Y_true, 'k', 'Linewidth', [2])
hold on
plot(year_f, Y_model_f, 'r--', 'Linewidth', [2])


%% definition of the function to optimize
function distance = lotv_dist(parameters)
    global u0
    global years_span
    global Y_true
    global X_true
    global year_v
        
    b = parameters(1);
    p = parameters(2);
    r = parameters(3);
    d = parameters(4);
    
    x0 = parameters(5);
    y0 = parameters(6);
    
    lv_stepper = @(t,u) lv(b, p, r, d, t, u);

    [t_out, u_out] = ode45(lv_stepper, [0 1], [x0; y0]);

    X_model_r = u_out(:,1);
    Y_model_r = u_out(:,2);
    year_r = year_v(1) + t_out * years_span;

    hi_fq_sample = linspace(year_v(1), year_v(end), 200);

    X_model_hf = interp1(year_r, X_model_r, hi_fq_sample);
    Y_model_hf = interp1(year_r, Y_model_r, hi_fq_sample);
    
    X_true_hf = interp1(year_v, X_true, hi_fq_sample);
    Y_true_hf = interp1(year_v, Y_true, hi_fq_sample);
    
    %% minimized quantity
    % the basic idea is to fit the average frequency and the average
    % height of the peaks
    [xm_pk, xm_locs] = findpeaks(X_model_hf, 'MinPeakProminence', 40);
    xm_avg = mean(xm_pk);
    xm_freq = mean(diff(xm_locs));
    [ym_pk, ym_locs] = findpeaks(Y_model_hf, 'MinPeakProminence', 10);
    ym_avg = mean(ym_pk);
    ym_freq = mean(diff(ym_locs));
    
    [xt_pk, xt_locs] = findpeaks(X_true_hf, 'MinPeakProminence', 40);
    xt_avg = mean(xt_pk);
    xt_freq = mean(diff(xt_locs));
    [yt_pk, yt_locs] = findpeaks(Y_true_hf, 'MinPeakProminence', 10);
    yt_avg = mean(yt_pk);
    yt_freq = mean(diff(yt_locs));
    
    distance = (xm_avg - xt_avg)^2 + (xm_freq - xt_freq)^2 + ... 
                (ym_avg - yt_avg)^2 + (ym_freq - yt_freq)^2;
    

    % distance = sum([abs(X_true_hf - X_model_hf), abs(Y_true_hf - Y_model_hf)]);
end