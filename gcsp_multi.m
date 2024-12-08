
close all

fs = 256;
n_classes = 5;
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
        
        A = zeros(n_channels, n_channels, n_classes);
        X_hat = zeros(N_data, ngft, L, n_classes);
        for c=1:n_classes
            indc = find(Y_train==c)';
            indnc = find(Y_train~=c)';

            Xc = [];
            Xnc = [];
    
            for i=indc
                Xc = [Xc, squeeze(Xt(i, :, :))];
            end
    
            for i=indnc
                Xnc = [Xnc, squeeze(Xt(i, :, :))];
            end
    
            a = 1;
            b = 1;
            gamma = 1.2;
            %gamma = 3;
    
            Zc = squareform(pdist([Xc, -gamma*Xnc], "correlation"));
            %Znc = squareform(pdist([Xnc, -gamma*Xc], "correlation"));
            
            Th = 0.0001;
            Ac = gsp_learn_graph_log_degrees(Zc, a, b);
            Ac(Ac < Th)=0;
            A(:, :, c) = Ac;
            %A1 = gsp_learn_graph_log_degrees(Z1, a, b);
            %A1(A1 < Th)=0;
            
    
            for i=1:N_data   
                X_hat(i, :, :, c) = gft_coef(squeeze(Xt(i, :, :)), Ac, ngft);
            end 
        end
        
        F = [];
        for c=1:n_classes
        
            Sc = zeros(size(X_hat, 2));
            Snc = zeros(size(X_hat, 2));
            nC = 0;
            nNC = 0;
    
            X_hat_train = X_hat(idx_train, :, :, :);  
            X_hat_val = X_hat(idx_val, :, :, :);
        
        
            % Calculating covaricance matrices
            for i=1:length(Y_train)
                class = Y_train(i);
                if class==c
                    Sc = Sc + squeeze(X_hat(i, :, :, c))*squeeze(X_hat(i, :, :, c))';
                    nC = nC + 1;
                else
                    Snc = Snc + squeeze(X_hat(i, :, :, c))*squeeze(X_hat(i, :, :, c))';
                    nNC = nNC + 1;
                end
            end
            Sc = Sc / nC;
            Snc = Snc / nNC;
                    
            % Getting weights by GEVD
            [Wc, D] = eig(Sc, Snc);
            [D,I] = sort(diag(D), 'descend');
            Wc = Wc(:, I);
            
            % CSP filters
            Wcsp = [Wc(:, 1:m)];
    
            % Filtering signals
            Xc_filtered = zeros(N_data, m, size(Xt, 3));
            
            for i=1:N_data
                Xc_filtered(i, :, :) = Wcsp'* squeeze(X_hat(i, :, :, c));
            end
            
            % Variance of rows as features
            F = [F, var(Xc_filtered, 1, 3)];
        end
        % Decision Tree Classifier
        Mdl = fitcensemble(F(idx_train, :), Y_train,'Method','Bag','NumLearningCycles',400,'Learners','tree');
        %Mdl = fitctree(F(idx_train, :), Y_train);
    
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
for i=1:n_classes
    subplot(1, n_classes, i)
    u = abs(A(:, :, i)-mean(A, 3));
    imshow(u/max(u(:)))
end

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


