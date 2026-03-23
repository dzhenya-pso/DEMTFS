% Wrapper Feature Selection Toolbox 
%---Input-------------------------------------------------------------
% feat   : Feature vector matrix (Instances x Features)
% label  : Label matrix (Instances x 1)
% opts   : Parameter settings 
% opts.N : Number of solutions / population size (* for all methods)
% opts.T : Maximum number of iterations (* for all methods)
% opts.k : Number of k in k-nearest neighbor 

%---Output------------------------------------------------------------
% FS    : Feature selection model (It contains several results)
% FS.sf : Index of selected features
% FS.ff : Selected features
% FS.nf : Number of selected features
% FS.c  : Convergence curve
% Acc   : Accuracy of validation model

clear, clc, close;
warning('off')
warning off all
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
opts.k = 5; % Number of k in K-nearest neighbor
ho = 0.3;% Ratio of validation data
% Parameters of PSO
opts.c1 = 2.5;
opts.c2 = 0.5;
opts.w  = 0.9;
opts.thres = 0.6;   % Threshold for selected features
runtimes = 20;      % Number of experimental runs


data_list = {"Colon"};

for p = 1:length(data_list)  % Iterate through all datasets
    data_name = data_list{p};
    dataname = fullfile("dataset", data_name + ".mat");  
    loaded_data = load(dataname);
    fprintf(">>>>>>>>>>load NO.%d data: <%s> \n", p, data_name);
    
    % Select the appropriate data assignment method based on the variable name
    if isfield(loaded_data, 'data')
        X = loaded_data.data(:, 2:end);
        Y = loaded_data.data(:, 1);
    elseif isfield(loaded_data, 'X') && isfield(loaded_data, 'Y')
        X = loaded_data.X;
        Y = loaded_data.Y;
    else
        error('Unexpected data format in %s', dataname);
    end
    % Map -1,1 to 0,1
    Y(Y==-1) = 0;
    feat = X;
    label = Y;
    
    for i = 1:runtimes
        fprintf('--------NO.%d starts running--------\n', i)
        
        % Divide data into training and validation sets
        HO = cvpartition(label, 'HoldOut', ho); 
        opts.Model = HO;
% ----------------Calculate selection probability using Lasso norm-------------------
        % Calculate standard deviation
        X_std = std(X);
        % Check if there are columns with zero standard deviation
        if all(X_std ~= 0)
            % No columns with zero standard deviation, perform normal normalization
            X_mean = mean(X);
            % Data normalization
            X_standardized = (X - X_mean) ./ X_std;
        else 
            X_standardized = X;
        end

        [B, FitInfo] = lasso(X_standardized, Y, 'CV', 10); % 10-fold cross-validation
        bestLambda = FitInfo.LambdaMinMSE; % Select Lambda that minimizes MSE
        bestB = B(:, FitInfo.IndexMinMSE); % Select corresponding regression coefficients
        % Calculate weight parameters via Lasso
        [B, FitInfo] = lasso(X_standardized, Y, 'Lambda', bestLambda); 
        % Calculate accuracy and feature count for Lasso method
        numFeatures = size(feat, 2); % Number of features
        Pos   = 1:numFeatures;
        Lasso_Matrix = double(B ~= 0);
        opts.Lasso = Lasso_Matrix';  % Transpose operation

% %---------------- Perform feature selection by different filter method ----------------
        FS_Filter = jffs('rf', feat, label, opts);
        FS_weight = FS_Filter.s;
		% Normalize to ensure no NaN or Inf values appear
        opts.weight = normalize(FS_weight, 'range');
        opts.weight(isnan(opts.weight)) = 1;
        opts.weight(isinf(opts.weight)) = 0;

        % Perform feature selection by DEMTFS
        FS = jfs('DEMTFS', feat, label, opts);
        fprintf('\n NO.%d: ACC= %f%%; FeaturesNumber=%d; Time=%d s \n', i, 100*FS.acc, FS.nf, FS.time);
    end
end






