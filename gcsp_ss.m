
close all

fs = 256;

y(y==class1) = 0;
y(y==class2) = 1;
n_channels = size(Xt, 2);
N_data = size(Xt, 1);
L = size(Xt, 3);
ngft = 14;

n_folds = 5;
number_of_filters = [6, 8, 10];

best_val_accuracy = 0;
m_filter_best = 0;

for m=number_of_filters
    average_accuracy = 0;
    average_accuracy_train = 0;

    for k=1:n_folds
        window = int32(N_data/n_folds);
        idx_val = 1+(k-1)*window:k*window;
        idx_train = 1:N_data;
        idx_train(idx_val) = [];

        Y_train = y(idx_train);
        Y_val = y(idx_val);
        
        ind0 = find(Y_train==0)';
        ind1 = find(Y_train==1)';

        X0 = [];
        X1 = [];

        for i=ind0
            X0 = [X0, squeeze(Xt(i, :, :))];
        end

        for i=ind1
            X1 = [X1, squeeze(Xt(i, :, :))];
        end

        a = 1;
        b = 1;
        gamma = 30000;
        %gamma = 3;

        Z0 = squareform(pdist([X0, -gamma*X1], "correlation"));
        Z1 = squareform(pdist([X1, -gamma*X0], "correlation"));
        
        Th = 0.0001;
        A0 = gsp_learn_graph_log_degrees(Z0, a, b);
        A0(A0 < Th)=0;
        A1 = gsp_learn_graph_log_degrees(Z1, a, b);
        A1(A1 < Th)=0;

        X_hat0 = zeros(N_data, ngft, L);
        X_hat1 = zeros(N_data, ngft, L);

        for i=1:N_data   
            X_hat0(i, :, :) = gft_coef(squeeze(Xt(i, :, :)), A0, ngft);
            X_hat1(i, :, :) = gft_coef(squeeze(Xt(i, :, :)), A1, ngft);
        end 

        C0 = zeros(size(X_hat0, 2));
        C1 = zeros(size(X_hat1, 2));

        X_hat0_train = X_hat0(idx_train, :, :);  
        X_hat0_val = X_hat0(idx_val, :, :);

        X_hat1_train = X_hat1(idx_train, :, :);  
        X_hat1_val = X_hat1(idx_val, :, :);


        % Calculating covaricance matrices
        for i=ind0
            C0 = C0 + squeeze(X_hat0(i, :, :))*squeeze(X_hat0(i, :, :))'/length(ind0);
        end
        
        for i=ind1
            C1 = C1 + squeeze(X_hat1(i, :, :))*squeeze(X_hat1(i, :, :))'/length(ind1);
        end
                
        % Getting weights by GEVD
        [W0, D] = eig(C0, C1);
        [D,I] = sort(diag(D), 'descend');
        W0 = W0(:, I);

        [W1, D] = eig(C1, C0);
        [D,I] = sort(diag(D), 'descend');
        W1 = W1(:, I);
        
        % CSP filters
        %Wcsp0 = W0(:, 1:m);
        %Wcsp1 = W1(:, 1:m); 
        Wcsp0 = [W0(:, 1:m/2), W0(:, end-m/2+1:end)];
        Wcsp1 = [W1(:, 1:m/2), W1(:, end-m/2+1:end)];

        % Filtering signals
        X0_filtered = zeros(N_data, m, size(Xt, 3));
        X1_filtered = zeros(N_data, m, size(Xt, 3));
        
        for i=1:N_data
            X0_filtered(i, :, :) = Wcsp0'* squeeze(X_hat0(i, :, :));
            X1_filtered(i, :, :) = Wcsp1'* squeeze(X_hat1(i, :, :));
        end
        
        % Variance of rows as features
        F = [var(X0_filtered, 1, 3), var(X1_filtered, 1, 3)];
        
        % Decision Tree Classifier
        Mdl = fitcensemble(F(idx_train, :), Y_train,'Method','Bag','NumLearningCycles',400,'Learners','tree');
        %Mdl = fitctree(F(idx_train, :), Y_train);
        %Mdl = fitcsvm(F(idx_train, :), Y_train);
    
        Y_pred = predict(Mdl,F(idx_val, :));
        accuracy = mean(Y_pred==Y_val);
        average_accuracy = average_accuracy + accuracy/n_folds;
        %accuracy_train = length(find(predict(Mdl,X_train) == y_train))/length(y_train);
        %average_accuracy_train = average_accuracy_train + accuracy_train/K_folds;
    
    end

    if average_accuracy > best_val_accuracy 
        best_val_accuracy = average_accuracy;
        m_filter_best = m;
    end
end
fprintf("Highest validation accuracy = %d \n", best_val_accuracy)
fprintf("Best number of filters = %d \n", m_filter_best)

%%

Ac = location_based_connectivity(electrodes);

figure 
subplot(1, 4, 1)
imshow(A0/max(A0(:)))
subplot(1, 4, 2)
imshow(A1/max(A1(:)))
subplot(1, 4, 3)
imshow(abs(A1-A0)/max((abs(A1(:)-A0(:)))))
subplot(1, 4, 4)
imshow(Ac/max(Ac(:)))

function [A, Z] = extract_connectivity(X, a, b, fs)
    %X = normalize(X, 1, "norm");
    Z = squareform(pdist(X, "correlation"));
   
%{
%}
    n_channels = size(X, 1);
    A = ones(n_channels);
    Pxx = [];
    for i=1:n_channels
        Pxx = [Pxx; pwelch(X(i, :), [], [], [], fs)];
    end
    for i=1:n_channels-1
        for j=i+1:n_channels
            Sxy = abs(cpsd(X(i, :), X(j, :), [], [], [], fs)).^2; 
            e = mean(Sxy./Pxx(i, :)./Pxx(j, :));
            Z(i, j) = e;
            Z(j, i) = e;
        end
    end

    A = gsp_learn_graph_log_degrees(Z, a, b);

    
end

function X_hat = gft_coef(X, A, ngft)
    D = diag (sum (A, 1)); % degree matrix
    L = D - A; % laplacian matrix
    
    [V,D] = eig(L);
    [D,I] = sort(diag(D), 'ascend');
    V = V(:, I(1:ngft));
    
    X_hat = V' * X;
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
    for i=1:N-1
         
    end
end
%%



