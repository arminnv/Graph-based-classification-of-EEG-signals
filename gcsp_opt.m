
close all

fs = 256;

y(y==class1) = 0;
y(y==class2) = 1;
n_channels = size(Xt, 2);
N_data = size(Xt, 1);
L = size(Xt, 3);

Xf = zeros(N_data, n_channels, L);
for i=1:N_data
    for ch=1:n_channels
        x = squeeze(Xt(i, ch, :));
        x_f = abs(fft(x));
        f = fs*(0:(length(x_f)/2))/length(x_f); % Frequency vector
        Xf(i, ch, :) = x_f;
    end
end

A0 = location_based_connectivity(electrodes);
A0(A0<0.0001)=0;

%a=1;
%b=0.7;
a = 1;
b = 0.7;
ngft = 14;

%X = Xt;
n_folds = 5;
%number_of_filters = 2*[1, 2, 3, 4, 5, 6];
number_of_filters = [10];

best_val_accuracy = 0;
m_filter_best = 0;
for m=number_of_filters
    average_accuracy = 0;
    average_accuracy_train = 0;

    for k=1:n_folds
        A = A0;
        
        
        window = int32(N_data/n_folds);
        idx_val = 1+(k-1)*window:k*window;
        idx_train = 1:N_data;
        idx_train(idx_val) = [];

        X_train = X(idx_train, :, :);
        Y_train = y(idx_train);
        X_val = X(idx_val, :, :);
        Y_val = y(idx_val);
        
        [model, F] = fit_model(Xt, Y_train, m, idx_train, A0, ngft);
    
        Y_pred = predict(model,F(idx_val, :));
        accuracy = mean(Y_pred==Y_val);
        average_accuracy = average_accuracy + accuracy/n_folds;
        %accuracy_train = length(find(predict(Mdl,X_train) == y_train))/length(y_train);
        %average_accuracy_train = average_accuracy_train + accuracy_train/K_folds;
        fprintf('accuracy %d : %d\n', k, accuracy)
    
    end

    if average_accuracy > best_val_accuracy 
        best_val_accuracy = average_accuracy;
        m_filter_best = m;
    end
end
fprintf("Highest validation accuracy = %d \n", best_val_accuracy)
fprintf("Best number of filters = %d \n", m_filter_best)


function [X_hat, TV]  = gft_coef(X, A, ngft)
    D = diag (sum (A, 1)); % degree matrix
    L = D - A; % laplacian matrix
    
    [V0,D] = eig(L);
    [D,I] = sort(diag(D), 'ascend');
    V = V0(:, I(1:ngft));
    
    X_hat = V' * X;

    TV = squeeze(D(1:ngft)'*(sum(X_hat.^2, 2)));

    %X_hat = normalize(X_hat, 1, "zscore");
end

function A = location_based_connectivity(electrodes)
    N = length(electrodes);
    A = zeros(N);
    for i=1:N-1
        for j=i+1:N
            v1 = [electrodes(i).X, electrodes(i).Y, electrodes(i).Z];
            v2 = [electrodes(j).X, electrodes(j).Y, electrodes(j).Z];
            A(i, j) = 1/norm(v1 - v2).^2;
            A(j, i) = A(i, j);
        end
    end
    A = A/max(A(:));
    %A = gsp_learn_graph_log_degrees(A, a, b);
end


function [model, F] = fit_model(Xt, Y_train, m, idx_train, A0, ngft)
    N_data = size(Xt, 1);
    L = size(Xt, 3);

    ind0 = find(Y_train==0)';
    ind1 = find(Y_train==1)';

    %options = optimoptions('fminunc','Display', 'iter');
    %options = optimoptions('fminunc','StepTolerance',0.01,'PlotFcn','pswplotbestf');

    %[A,fval] = fminunc(@(x) loss_fn(x, Xt, Y_train, m), A0, [], [], [], [], 0, 1, [], options);
    swarm_size = 50;
    [U, S, V] = svd(A0);
    %nvars = 7; %8;
    nvars = length(A0(:));
    X0 = zeros(swarm_size, nvars);
    S = diag(S);
    
    for i=1:swarm_size
        %X0(i, :) = S(1:nvars);
        X0(i, :) = A0(:);
    end

    options = optimoptions('particleswarm','SwarmSize',swarm_size,"MaxIterations",20 ...
   ,"SocialAdjustmentWeight",0.8,"SelfAdjustmentWeight",0.8 ...
   ,"MaxStallIterations",10,"InitialSwarmMatrix", X0,'PlotFcn','pswplotbestf');

    rng default  % For reproducibility
    
    [x_ans,fval] = particleswarm(@(x) loss_fn(x, Xt(idx_train, :, :), Y_train, m, A0, ngft),nvars,zeros(nvars,1),1*ones(nvars,1),options);
    A = reshape(x_ans, size(A0));
    %disp(x_ans)
    
    %S(1:length(x_ans)) = x_ans;
    %A = U * diag(S) * V';
    %A = (A+A0)/2;
    A = (A+A')/2;
    A = A - diag(diag(A));
    A = gsp_learn_graph_log_degrees(A, 1, 0.5);
    A(A<0.001)=0;
    
    
    X_hat = zeros(N_data, ngft, L);
        for i=1:N_data
            [X_hat(i, :, :), TV] = gft_coef(squeeze(Xt(i, :, :)), A, ngft);
            
        end
        
    X = X_hat;

    
    % Calculating covaricance matrices
    
    C0 = zeros(size(X, 2));
    C1 = zeros(size(X, 2));
    
    for i=ind0
        C0 = C0 + squeeze(X(i, :, :))*squeeze(X(i, :, :))'/length(ind0);
    end
    
    for i=ind1
        C1 = C1 + squeeze(X(i, :, :))*squeeze(X(i, :, :))'/length(ind1);
    end
            
    % Getting weights by GEVD
    [W, D] = eig(C0, C1);
    

    [D,I] = sort(diag(D), 'descend');
    W = W(:, I);
    
    % CSP filters
    Wcsp = [W(:, 1:m/2), W(:, end-m/2+1:end)];
    
    % Filtering signals
    X_filtered = zeros(N_data, m, size(X, 3));
    
    for i=1:N_data
        X_filtered(i, :, :) = Wcsp'* squeeze(X(i, :, :));
    end
    
    % Variance of rows as features
    F = var(X_filtered, 1, 3);
    
    % Decision Tree Classifier
    model = fitcensemble(F(idx_train, :), Y_train,'Method','Bag','NumLearningCycles',400,'Learners','tree');
    %model = fitctree(F(idx_train, :), Y_train);
    %model = fitcdiscr(F(idx_train, :), Y_train);
end

function loss = loss_fn(A, Xt_train, Y_train, m, A0, ngft)
    
    nch = length(A0);
    
    %[U, S0, V] = svd(A0);
    %S0 = diag(S0);
    %S0(1:length(S)) = S;
    %A = U * diag(S0) * V';
    A = reshape(A, [nch, nch]);
    
    A = A + 0.01*randn(size(A0, 1));
    A = (A+A')/2;
    %A = gsp_learn_graph_log_degrees(A, 1, 1);
    %A = (A+A0)/2;
    A(A<0.001)=0;

    %A = reshape(A, [nch, nch]);
    %A = (A+A0)/2;
    A(A<0.001)=0;
    N_data = size(Xt_train, 1);
    
    L = size(Xt_train, 3);
    smoothness = 0;
    X_hat = zeros(N_data, ngft, L);
        for i=1:N_data
            [X_hat(i, :, :), TV] = gft_coef(squeeze(Xt_train(i, :, :)), A, ngft);
            smoothness = smoothness - TV/10^8;
        end
        
    X = X_hat;

    % Calculating covaricance matrices
    ind0 = find(Y_train(int32(N_data/2):end)==0)';
    ind1 = find(Y_train(int32(N_data/2):end)==1)';
    C0 = zeros(size(X, 2));
    C1 = zeros(size(X, 2));
    
    for i=ind0
        C0 = C0 + squeeze(X(i, :, :))*squeeze(X(i, :, :))'/length(ind0);
    end
    
    for i=ind1
        C1 = C1 + squeeze(X(i, :, :))*squeeze(X(i, :, :))'/length(ind1);
    end
            
    % Getting weights by GEVD
    [W, D] = eig(C0, C1);

    [D,I] = sort(diag(D), 'descend');
    W = W(:, I);
    
    % CSP filters
    Wcsp = [W(:, 1:m/2), W(:, end-m/2+1:end)];
    
    % Filtering signals
    X_filtered = zeros(N_data, m, size(X, 3));
    
    for i=1:N_data
        X_filtered(i, :, :) = Wcsp'* squeeze(X(i, :, :));
    end
    
    % Variance of rows as features
    F = var(X_filtered, 1, 3);

    mu0 = mean(F(ind0));
    mu1 = mean(F(ind1));
    %{
    smoothness = 0;
    D = diag (sum (A, 1)); % degree matrix
    Lp = D - A; % laplacian matrix
    for i=1:N_data 
        for t=1:L
            x = squeeze(Xt_train(i, :, t))';
            %disp(size(x))
            smoothness = smoothness + x'*Lp*x;
        end
    end
    smoothness = smoothness / N_data/L/nch;
    %}
   
    model = fitctree(F(int32(N_data/2):end, :), Y_train(int32(N_data/2):end));
    Y_pred = predict(model,F(1:int32(N_data/2), :));
    accuracy = mean(Y_pred==Y_train(1:int32(N_data/2)));
    %loss = sum((mu1 - mu0).^2./(var(F(ind0)) + var(F(ind1))));

    %loss = -10*accuracy - 2*smoothness - 4*sum((mu1 - mu0).^2./(var(F(ind0)) + var(F(ind1))));
    loss = -accuracy;
    %var(F(ind1))));
    %loss = -accuracy;
    D = diag (sum (A, 1)); % degree matrix
    
    %loss = -sum((mu1 - mu0).^2./(var(F(ind0)) + var(F(ind1)))) ;%+ sum(A(:)/max(A(:)))/50;
end