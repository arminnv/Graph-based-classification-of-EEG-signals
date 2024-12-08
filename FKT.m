fs = 256;  

channels = size(X, 3);
n = size(X, 1);
% Hyperparameters = [a, b, m]
%hp = [1, 1, 8];
global hp;
hp = [1, 1, 6];

n_frequency = 50;
df = 1/n_frequency;
Xf = zeros(size(X,1), n_frequency, channels);
for i=1:size(X,1)
    for ch=1:size(X,3)
        x = X(i, :, ch);
        N = 2^nextpow2(length(x)); % Number of FFT points
        x_f = fft(x,N)/length(x); % FFT normalized by the length of signal
       
        f = fs*(0:(N/2))/N; % Frequency vector
        i_start = find(f>=0.5);
        i_end = find(f>50);
        xf = zeros(n_frequency, 1);
        for k=1:n_frequency
            xf(k) = mean(x_f(i_start+(k-1)*round(N/fs): i_end+k*round(N/fs)));
        end
        Xf(i, :, ch) = abs(xf);
    end
end


Y = zeros(size(Xf));
labels = zeros(size(X, 1));
S = zeros(size(Xf, 1), size(Xf, 2), size(Xf, 2));
for i=1:size(Xf, 1)
    A = extract_connectivity(squeeze(Xf(i, :, :)));
    [S(i, :, :), Y(i, :, :)] = cov_matrix(squeeze(Xf(i, :, :)), A);
end

average_accuracy = 0;
average_accuracy_train = 0;
K_folds = 11;
for k=1:K_folds
    S_A = 0;
    S_N = 0;
    nA = 0;
    nN = 0;
    n_test = int32(n/K_folds);
    n_start = 1 + (k-1)*n_test; 
    n_end = k*n_test;
    for i=1:n
        if i>=n_start && i<=n_end
            continue
        end
        if y(i)==class1 
            S_A = S_A + squeeze(S(i, :, :));
            nA = nA + 1;
        elseif y(i)==class2 
            S_N = S_N + squeeze(S(i, :, :));
            nN = nN + 1;
        end
    end

    P = calculate_P(S_A, S_N, nA, nN);
    
    Z = zeros(size(Y));
    for i=1:length(y)
        Z(i, :, :) = P*squeeze(Y(i, :, :));
    end 
    
    Vars = [];
    m = 9;
    for sample=1:length(y)
        for row=[2:m+1  size(Z,2)-m+1:size(Z,2)]
            Vars(sample, row-1) = var(Z(sample, row, :));
        end
    end
    
    indexes = n_start:n_end;
    y_test = y(indexes);
    X_test = Vars(indexes, :);
    y_train = y;
    X_train = Vars;
    y_train(indexes) = [];
    X_train(indexes, :) = [];
    
    
    Mdl = fitctree(X_train,y_train);
    
    y_pred = predict(Mdl,X_test);
    accuracy = length(find(y_pred == y_test))/length(y_test);
    average_accuracy = average_accuracy + accuracy/K_folds;
    accuracy_train = length(find(predict(Mdl,X_train) == y_train))/length(y_train);
    average_accuracy_train = average_accuracy_train + accuracy_train/K_folds;
end


disp(average_accuracy)
disp(average_accuracy_train)


function P = calculate_P(S_A, S_N, nA, nN)
    nT = nA + nN;
    S_A = S_A/nA;
    S_N = S_N/nN;
    aA = nA/nT;
    aN = nN/nT;
    Savg = aA*S_A + aN*S_N;
    [Q,L] = eig_sorted(Savg);
    L(1:find(L<=1e-6, 1, 'last')) = 1;
    Gamma = L.^(-1/2);
    Q2 = diag(Gamma)*Q';
    
    S_A1 = Q2*S_A*Q2';
    S_N1 = Q2*S_N*Q2';
    
    ind = 1;
    for i=1:length(S_A1)
        if S_A1(i, i)>1e-8
            ind = i;
            break
        end
    end
    [T,S_A2] = eig_sorted(S_A1(ind:end, ind:end));
    
    S_A2 = diag(S_A2);
    
    T2 = eye(size(S_A1));
    T2(ind:end, ind:end)=T;
    %S_A2 = T2'*S_A1*T2;
    S_N2 = T2'*S_N1*T2;
    P = T2'*Q2;
end


function A = extract_connectivity(X)
    Dist = squareform(pdist(X));
    global hp;
    a = hp(1);
    b = hp(2);
    %A = gsp_learn_graph_log_degrees(Dist, a, b);
    A = corrcoef(X');
end

function [S, Y] = cov_matrix(X, A)
    D = diag (sum (A)); % degree matrix
    L = D - A; % laplacian matrix
    
    [V,D] = eig(L);
    [D,I] = sort(diag(D), 'ascend');
    V = V(:, I);
    
    X_hat = V' * X;
    Y = zeros(size(X_hat));
    for i=1:size(X_hat, 2)
        Y(:, i) = normalize(X_hat(:, i));
    end
    
    S = Y*Y';
    S = S/trace(S); % proxy of the sample covariance matrix
end

function [V, D] = eig_sorted(X)
    [V,D] = eig(X);
    [D,I] = sort(diag(D), 'ascend');
    V = V(:, I);
end