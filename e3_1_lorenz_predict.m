clear all; close all; clc

% train or test
mode = 'roll';

% rho values
rhos_train=[10, 28, 40];
rhos_test = [17, 35];

% fixed data
t=0:0.01:20;
sigma=10; b=8/3;
x0=[5 5 5];

% initializaton
samples=zeros(0);
targets = zeros(0);

switch mode
    
    %% train the network on the training rho values-------------------
    case 'train'
        % prepare the data
        for r = 1:3
            rho = rhos_train(r);
            [t,xsol]=ode45('lor_rhs',t,x0,[],sigma,b,rho);

            x_true=xsol(:,1);
            y_true=xsol(:,2);
            z_true=xsol(:,3);

            figure(1)
            subplot(1,3,r)
            plot3(x_true,y_true,z_true,'Linewidth', [2]);
            xlabel('x')
            ylabel('y')
            grid on

            samples = [samples; xsol(1:end-1,:)];
            targets = [targets; xsol(1+1: end,:)];

        end

        % do the actual training
        data_mtx = samples';
        target_v = targets';
        hiddenLayerSize = [9,3];
        nn_train_regr

        save('saved_nets/lornet_1.mat', 'net')
        
     %% train the network on the training rho values
    case 'test'
        % load a previously trained network
        load('saved_nets/lornet_1.mat')
        
        % prepare the test data and predict
        for r = 1:2
            rho = rhos_test(r);
            [t,xsol]=ode45('lor_rhs',t,x0,[],sigma,b,rho);

            x_true=xsol(:,1);
            y_true=xsol(:,2);
            z_true=xsol(:,3);

            predicted = net(xsol(1:end-1,:)')';
            x_pre=predicted(:,1);
            y_pre=predicted(:,2);
            z_pre=predicted(:,3);
            
            figure(2)
            subplot(1,2,r)

            plot3(x_true,y_true,z_true, 'k-', 'Linewidth', [1]);
            hold on
            plot3(x_pre,y_pre,z_pre, 'r.', 'Linewidth', [1], 'MarkerSize',10);
            xlabel('x')
            ylabel('y')
            zlabel('z')
            title(['Lorenz NN predict \rho=', num2str(rho)])
            grid on

        end
        
    case 'roll'
        t=0:0.01:5;
        % load a previously trained network
        load('saved_nets/lornet_1.mat')
        for r = 1:2
            
            rho = rhos_test(r);
            [t,xsol]=ode45('lor_rhs',t,x0,[],sigma,b,rho);
            x_true=xsol(:,1);
            y_true=xsol(:,2);
            z_true=xsol(:,3);
            
            x = xsol(100,:)';
            predicted = x;
            
            % roll the prediction
            for attempts = 1:100
                pre = net(x);
                x = pre;
                predicted = [predicted, x];
            end
            
            predicted = predicted';
            
            x_pre=predicted(:,1);
            y_pre=predicted(:,2);
            z_pre=predicted(:,3);
            
            figure(2)
            subplot(1,2,r)

            plot3(x_true,y_true,z_true, 'k-', 'Linewidth', [1]);
            hold on
            plot3(x_pre,y_pre,z_pre, 'r.-', 'Linewidth', [1], 'MarkerSize',10);
            xlabel('x')
            ylabel('y')
            zlabel('z')
            title(['Lorenz NN rolling predict \rho=', num2str(rho)])
            grid on
                
            
        end
end
