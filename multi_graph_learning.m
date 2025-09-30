class1_idx = (y==class1);
class2_idx = (y==class2);

y(class1_idx) = 0;
y(class2_idx) = 1;

disp('*')

%%

N_data = size(Xt, 1);
n_folds = 10;

gamma_values = [4, 3, 2, 1, 0.4, 0.35, 0.3, 0.25, 0.2];
number_of_filters = [24, 22, 20, 18, 16];
ngft_values = [14, 15, 16, 17];

best_val_accuracy = 0;
m_filter_best = 0;

for gamma = gamma_values
    disp(gamma)
for ngft = ngft_values
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

        Xt_train = (Xt(idx_train, :, :));
        Xt_val = (Xt(idx_val, :, :));
        
        [Mdl, A0, A1, Wcsp0, Wcsp1] = train_model(Xt_train, Y_train, gamma, ngft, m);
        accuracy = validate_model(Mdl, Xt_val, Y_val, A0, A1, Wcsp0, Wcsp1, ngft, m);
    
        
        average_accuracy = average_accuracy + accuracy/n_folds;
        %average_accuracy_train = average_accuracy_train + accuracy_train/K_folds;
    
    end

    if average_accuracy > best_val_accuracy 
        best_val_accuracy = average_accuracy;
        m_filter_best = m;

        subject = subjects{1};
        outputFilename =  "results/Ours2/" + subject + num2str(class1) + num2str(class2) + "_results.mat";
        save(outputFilename, 'subject', 'best_val_accuracy', 'gamma', ...
           'm_filter_best', 'A0', 'A1', 'Wcsp0', ...
           'Wcsp1', 'Mdl', 'ngft')
    end

end
end
end


fprintf("Highest validation accuracy = %d \n", best_val_accuracy)
fprintf("Best number of filters = %d \n", m_filter_best)




function [Mdl, A0, A1, Wcsp0, Wcsp1] = train_model(Xt_train, Y_train, gamma, ngft, m)
        N_train = length(Y_train);
        L = size(Xt_train, 3);

        ind0 = find(Y_train==0)';
        ind1 = find(Y_train==1)';

        X0 = [];
        X1 = [];

        for i=ind0
            X0 = [X0, squeeze(Xt_train(i, :, :))];
        end

        for i=ind1
            X1 = [X1, squeeze(Xt_train(i, :, :))];
        end

        a = 1;
        b = 1;

        Z0 = squareform(pdist_complex([X0, -1j*gamma*X1]));
        Z1 = squareform(pdist_complex([X1, -1j*gamma*X0]));
        
        Th = 0.0001;
        A0 = gsp_learn_graph_log_degrees(Z0, a, b);
        A0(A0 < Th)=0;
        A1 = gsp_learn_graph_log_degrees(Z1, a, b);
        A1(A1 < Th)=0;

        X_hat0 = zeros(N_train, ngft, L);
        X_hat1 = zeros(N_train, ngft, L);

        for i=1:N_train   
            X_hat0(i, :, :) = gft_coef(squeeze(Xt_train(i, :, :)), A0, ngft);
            X_hat1(i, :, :) = gft_coef(squeeze(Xt_train(i, :, :)), A1, ngft);
        end 

        C0 = zeros(size(X_hat0, 2));
        C1 = zeros(size(X_hat1, 2));

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
        X0_filtered = zeros(N_train, m, L);
        X1_filtered = zeros(N_train, m, L);
        
        for i=1:N_train
            X0_filtered(i, :, :) = Wcsp0'* squeeze(X_hat0(i, :, :));
            X1_filtered(i, :, :) = Wcsp1'* squeeze(X_hat1(i, :, :));
        end
        
        % Variance of rows as features
        F = [var(X0_filtered, 1, 3), var(X1_filtered, 1, 3)];
        
        % Decision Tree Classifier

        Mdl = fitcensemble(F, Y_train,'Method','Bag','NumLearningCycles',400,'Learners','tree');
        %Mdl = fitctree(F, Y_train);
        %Mdl = fitcsvm(F, Y_train);

end


function accuracy = validate_model(Mdl, Xt_val, Y_val, A0, A1, Wcsp0, Wcsp1, ngft, m)
    N_val = length(Y_val);
    L = size(Xt_val, 3);

    X_hat0 = zeros(N_val, ngft, L);
    X_hat1 = zeros(N_val, ngft, L);
    
    for i=1:N_val  
        X_hat0(i, :, :) = gft_coef(squeeze(Xt_val(i, :, :)), A0, ngft);
        X_hat1(i, :, :) = gft_coef(squeeze(Xt_val(i, :, :)), A1, ngft);
    end 

    X0_filtered = zeros(N_val, m, L);
    X1_filtered = zeros(N_val, m, L);
        
    for i=1:N_val
        X0_filtered(i, :, :) = Wcsp0'* squeeze(X_hat0(i, :, :));
        X1_filtered(i, :, :) = Wcsp1'* squeeze(X_hat1(i, :, :));
    end

    % Variance of rows as features
    F = [var(X0_filtered, 1, 3), var(X1_filtered, 1, 3)];

    Y_pred = predict(Mdl, F);
    accuracy = mean(Y_pred==Y_val);
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


function d = pdist_complex(X)
% Compute correlation distance between complex-valued rows of X
% Input:  X — n x d complex matrix
% Output: d — condensed distance vector, like pdist(X, 'correlation')

    [n, ~] = size(X);
    m = n * (n - 1) / 2;
    d = zeros(1, m);

    idx = 1;
    for i = 1:n-1
        xi = X(i, :) - mean(X(i, :));
        for j = i+1:n
            xj = X(j, :) - mean(X(j, :));
            
            num = real(xi * xj');  % use real part to avoid complex distance
            denom = norm(xi) * norm(xj);
            if denom == 0
                dist = 1;  % fallback for zero vectors
            else
                dist = 1 - num / denom;
            end

            d(idx) = dist;
            idx = idx + 1;
        end
    end
end





