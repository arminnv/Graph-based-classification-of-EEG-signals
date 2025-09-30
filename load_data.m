subjects = {'D'}; %"{'A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L'}"
sessions = [1]; %, 2]
fs = 256;
t_start = int32(4.5*fs);    %6
t_end = int32(7.5*fs);      %10 

exclusions = [];


class1 = 2;
class2 = 5;

% Delta Theta Alpha Beta Gamma
fbands = struct('Delta', [0.5, 4], ...
                'Theta', [4, 8], ...
                'Alpha', [8, 13], ...
                'Beta', [13, 35], ...
                'Gamma', [35, 100]);

electrode_names = ["AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", ...
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1",...
    "Pz", "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2"];

electrodes = [];
for j=1:length(electrode_names)
    for i=1:length(AllElectrodes)
        ind = cell2mat(strfind(electrode_names, AllElectrodes(i).labels));
        if strcmp(electrode_names(j), AllElectrodes(i).labels)
            electrodes = [electrodes, AllElectrodes(i)];
            break
        end
    end
end

electrodes(exclusions) = [];

X = [];
labels = [];

for subject = subjects
    for session = sessions
        data = load(subject + ".mat").data;
        data = data{1, session};
        trial = int32(data.trial);
        y = int32(data.y);

        %x = data.X;
        %x(:, exclusions) = [];

        x = data.X;
        sigma = std(x, 0, 1);
        x = x ./ sigma;
   
        % Splitting trials
        for i=1:length(trial)
            
                if ~(y(i)==class1 || y(i)==class2)
                    continue
                end 
            
            X(end+1, :, :) = x(trial(i) + t_start-1: trial(i) + t_end, :);
            %X(end, :, :) = bandpass(squeeze(X(end, :, :)), [0.5, 50], fs);
            labels(end+1) = y(i);
        end
    end
end


y = labels';
%X = cell2mat(X);
%y = cell2mat(y);
disp(size(y));
Xt = permute(X, [1, 3, 2]);




%A = cell2mat(A);
%A, y = shuffle(A, y);
%disp("A shape: ", size(A));