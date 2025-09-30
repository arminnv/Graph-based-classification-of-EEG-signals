class1_idx = (y==class1);
class2_idx = (y==class2);

y(class1_idx) = 0;
y(class2_idx) = 1;


%%

N_data = size(Xt, 1);
n_folds = 10;

number_of_filters = [24, 22, 20, 18, 16]; % [16, 14, 12, 10, 8, 6, 4];

best_val_accuracy = 0;
m_filter_best = 0;

for m = number_of_filters

    average_accuracy = 0;
    average_accuracy_train = 0;

    for k=1:n_folds
        window = int32(N_data/n_folds);
        idx_val = 1+(k-1)*window: k*window;
        idx_train = 1:N_data;
        idx_train(idx_val) = [];

        Y_train = y(idx_train);
        Y_val = y(idx_val);

        Xt_train = (Xt(idx_train, :, :));
        Xt_val = (Xt(idx_val, :, :));
        
        [Mdl, Wcsp] = train_model(Xt_train, Y_train, m);
        accuracy = validate_model(Mdl, Xt_val, Y_val, Wcsp, m);
    
        average_accuracy = average_accuracy + accuracy/n_folds;
        %average_accuracy_train = average_accuracy_train + accuracy_train/K_folds;
    
    end

    if average_accuracy > best_val_accuracy 
        best_val_accuracy = average_accuracy;
        m_filter_best = m;
        disp(best_val_accuracy)

        subject = subjects{1};
        outputFilename =  "results/CSP2/" + subject + num2str(class1) + num2str(class2) + "_results.mat";
        save(outputFilename, 'subject', 'best_val_accuracy', ...
           'm_filter_best', 'Wcsp', 'Mdl')
    end

end


fprintf("Highest validation accuracy = %d \n", best_val_accuracy)
fprintf("Best number of filters = %d \n", m_filter_best)


function [Mdl, Wcsp] = train_model(Xt_train, Y_train, m)
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

        C0 = zeros(size(Xt_train, 2));
        C1 = zeros(size(Xt_train, 2));

        % Calculating covaricance matrices
        for i=ind0
            C0 = C0 + squeeze(Xt_train(i, :, :))*squeeze(Xt_train(i, :, :))'/length(ind0);
        end
        
        for i=ind1
            C1 = C1 + squeeze(Xt_train(i, :, :))*squeeze(Xt_train(i, :, :))'/length(ind1);
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
            X_filtered(i, :, :) = Wcsp'* squeeze(Xt_train(i, :, :));
        end
        
        % Variance of rows as features
        F = var(X_filtered, 1, 3);
        
        % Decision Tree Classifier

        Mdl = fitcensemble(F, Y_train,'Method','Bag','NumLearningCycles',400,'Learners','tree');
        %Mdl = fitctree(F, Y_train);
        %Mdl = fitcsvm(F, Y_train);

end


function accuracy = validate_model(Mdl, Xt_val, Y_val, Wcsp, m)
    N_val = length(Y_val);
    L = size(Xt_val, 3);

    X_filtered = zeros(N_val, m, L);
        
    for i=1:N_val
        X_filtered(i, :, :) = Wcsp'* squeeze(Xt_val(i, :, :));
    end

    % Variance of rows as features
    F = var(X_filtered, 1, 3);

    Y_pred = predict(Mdl, F);
    accuracy = mean(Y_pred==Y_val);
end





