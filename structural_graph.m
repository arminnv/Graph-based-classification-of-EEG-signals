class1_idx = (y==class1);
class2_idx = (y==class2);

y(class1_idx) = 0;
y(class2_idx) = 1;


%%

N_data = size(Xt, 1);
n_folds = 10;

A = location_based_connectivity(electrodes);

%number_of_filters = [4, 6, 8, 10, 12, 14, 16];
number_of_filters = [16, 14, 12, 10, 8, 6];
ngft_values = [16, 15, 14];

best_val_accuracy = 0;
m_filter_best = 0;

for m = number_of_filters
for ngft = ngft_values


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
        
        [Mdl, Wcsp] = train_model(Xt_train, Y_train, A, ngft, m);
        accuracy = validate_model(Mdl, Xt_val, Y_val, A, Wcsp, ngft, m);
    
        
        average_accuracy = average_accuracy + accuracy/n_folds;
        %average_accuracy_train = average_accuracy_train + accuracy_train/K_folds;
    
    end

    if average_accuracy > best_val_accuracy 
        best_val_accuracy = average_accuracy;
        m_filter_best = m;
        disp(best_val_accuracy)

        subject = subjects{1};
        outputFilename =  "results/Structural/" + subject + num2str(class1) + num2str(class2) + "_results.mat";
        save(outputFilename, 'subject', 'best_val_accuracy', ...
           'm_filter_best', 'Wcsp', ...
           'Mdl', 'ngft')
    end

end
end


fprintf("Highest validation accuracy = %d \n", best_val_accuracy)
fprintf("Best number of filters = %d \n", m_filter_best)




function [Mdl, Wcsp] = train_model(Xt_train, Y_train, A, ngft, m)
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


        

        X_hat = zeros(N_train, ngft, L);
        

        for i=1:N_train   
            X_hat(i, :, :) = gft_coef(squeeze(Xt_train(i, :, :)), A, ngft);
        end 

        C0 = zeros(size(X_hat, 2));
        C1 = zeros(size(X_hat, 2));

        % Calculating covaricance matrices
        for i=ind0
            C0 = C0 + squeeze(X_hat(i, :, :))*squeeze(X_hat(i, :, :))'/length(ind0);
        end
        
        for i=ind1
            C1 = C1 + squeeze(X_hat(i, :, :))*squeeze(X_hat(i, :, :))'/length(ind1);
        end
                
        % Getting weights by GEVD
        [W, D] = eig(C0, C1);
        [D,I] = sort(diag(D), 'descend');
        W = W(:, I);
        
        % CSP filters
        %Wcsp0 = W0(:, 1:m);
        %Wcsp1 = W1(:, 1:m); 
        Wcsp = [W(:, 1:m/2), W(:, end-m/2+1:end)];

        % Filtering signals
        X_filtered = zeros(N_train, m, L);
        
        for i=1:N_train
            X_filtered(i, :, :) = Wcsp'* squeeze(X_hat(i, :, :));
        end
        
        % Variance of rows as features
        F = var(X_filtered, 1, 3);
        
        % Decision Tree Classifier

        Mdl = fitcensemble(F, Y_train,'Method','Bag','NumLearningCycles',400,'Learners','tree');
        %Mdl = fitctree(F, Y_train);
        %Mdl = fitcsvm(F, Y_train);

end


function accuracy = validate_model(Mdl, Xt_val, Y_val, A, Wcsp, ngft, m)
    N_val = length(Y_val);
    L = size(Xt_val, 3);

    X_hat = zeros(N_val, ngft, L);
    
    for i=1:N_val  
        X_hat(i, :, :) = gft_coef(squeeze(Xt_val(i, :, :)), A, ngft);
    end 

    X_filtered = zeros(N_val, m, L);
        
    for i=1:N_val
        X_filtered(i, :, :) = Wcsp'* squeeze(X_hat(i, :, :));
    end

    % Variance of rows as features
    F = var(X_filtered, 1, 3);

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





