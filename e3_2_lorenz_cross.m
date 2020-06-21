clear all; close all; clc

% train or test
mode = 'test';

% rho values
rhos_train=[10, 28, 40];
rhos_test = [17, 35];

% fixed data
t=0:0.01:2000;
sigma=10; b=8/3;
x0=[5 5 5];

% initializaton
switch mode
    
    %% train the network on the training rho values-------------------
    case 'train'
        % prepare the data

        rho = 28;
        t=0:0.01:2000;  
        [t,xsol]=ode45('lor_rhs',t,x0,[],sigma,b,rho);
        x_true=xsol(:,1);
        gonna_cross0 = double((sign(x_true(2:end)) - sign(x_true(1:end-1))) ~= 0);
        
        % this event is just too rare for training, we'll flag as
        % "imminent" also a few steps before the trainsition
        extension = 10;
        gonna_cross = gonna_cross0;
        for e = 2:extension
            gonna_cross = gonna_cross + [gonna_cross0(e:end); zeros(e-1,1)];
        end
        
        samples = xsol(1:end-1,:)';
        targets = gonna_cross;
        
        % 1-hot encoding
        targets = [targets, 1-targets]';
        
        % do the actual training
        %% WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        % for some reason this does not train in the same way as using 
        % the GUI tool
        nn_train_cross
        %save('lorcross_1.mat', 'net')
        
     %% train the network on the training rho values
    case 'test'
        % load a previously trained network
        load('lorcross_1.mat')
        % prepare the test data and predict
        rho = 28;
        t=0:0.01:20;
        [t,xsol]=ode45('lor_rhs',t,x0,[],sigma,b,rho);

        x_true=xsol(:,1);
        y_true=xsol(:,2);
        z_true=xsol(:,3);

        predicted = net(xsol(1:end,:)')';
        
        % plot the predictions directly on the Lorenz trajectory
        figure(2)
        colormap(winter)
        dsize = predicted(:,1) * 20 + 5;
        color = predicted(:,1);
        scatter3(x_true,y_true,z_true,dsize, color, 'o', 'filled');
        title('Lorenz system - Predicting lobe transition')
        xlabel('x')
        ylabel('y')
        grid on
        c = colorbar();
        c.Label.String = 'Est. P of imminent transition';

end