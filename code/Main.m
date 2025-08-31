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

for p = 1:length(data_list)  % 遍历所有数据集
    data_name = data_list{p};
    dataname = fullfile("dataset", data_name + ".mat");  
    loaded_data = load(dataname);
    fprintf(">>>>>>>>>>load NO.%d data: <%s> \n", p, data_name);
    
    % 根据变量名选择合适的数据赋值方式
    if isfield(loaded_data, 'data')
        X = loaded_data.data(:, 2:end);
        Y = loaded_data.data(:, 1);
    elseif isfield(loaded_data, 'X') && isfield(loaded_data, 'Y')
        X = loaded_data.X;
        Y = loaded_data.Y;
    else
        error('Unexpected data format in %s', dataname);
    end
    % 将 -1,1 映射为 0,1
    Y(Y==-1) = 0;
    feat = X;
    label = Y;
    
    for i = 1:runtimes
        fprintf('--------NO.%d starts running--------\n', i)
        
        % Divide data into training and validation sets
        HO = cvpartition(label, 'HoldOut', ho); 
        opts.Model = HO;
% ----------------利用Lasso范数计算选择概率-------------------
        % 计算标准差
        X_std = std(X);
        % 判断是否存在标准差为0的列
        if all(X_std ~= 0)
            % 没有标准差为0的列，正常标准化
            X_mean = mean(X);
            % 数据标准化
            X_standardized = (X - X_mean) ./ X_std;
        else 
            X_standardized = X;
        end

        [B, FitInfo] = lasso(X_standardized, Y, 'CV', 10); % 10 折交叉验证
        bestLambda = FitInfo.LambdaMinMSE; % 选取使 MSE 最小的 Lambda
        bestB = B(:, FitInfo.IndexMinMSE); % 选取对应的回归系数
        % 通过Lasso计算权重参数
        [B, FitInfo] = lasso(X_standardized, Y, 'Lambda', bestLambda); 
        % 计算Lasso method 的准确率和特征数量
        numFeatures = size(feat, 2); % 特征数量
        Pos   = 1:numFeatures;
        Lasso_Matrix = double(B ~= 0);
        opts.Lasso = Lasso_Matrix';  % 转置操作

% %---------------- Perform feature selection by different filter method ----------------
        FS_Filter = jffs('rf', feat, label, opts);
        FS_weight = FS_Filter.s;
        opts.weight = normalize(FS_weight, 'range');
        % 归一化保证不能出现NaN和Inf
        opts.weight(isnan(opts.weight)) = 1;
        opts.weight(isinf(opts.weight)) = 0;

        % Perform feature selection by DEMTFS
        FS = jfs('DEMTFS', feat, label, opts);
        fprintf('\n NO.%d: ACC= %f%%; FeaturesNumber=%d; Time=%d s \n', i, 100*FS.acc, FS.nf, FS.time);
    end
end






