% Task creation through FFT  (2/22/2025)

% == == == == == == == == Input == == == == == == == == %
% NC  : 特征数量是否变化                                %
% AC  : 准确率是否变化                                  %
% feat  : 特征                                          %
% label : 标签(类别)                                    %
% X     : 连续编码方式，>0.5即选择特征为1，反之不选此特征 %
% opts  : 进化算法的参数结构体                           %
% == == == == == == == == == == == == == == == == == == %

% == == == == == == == == Output == == == == == == == == %
% task  : 处理后的任务                                   %
% == == == == == == == == == == == == == == == == == == %
function PSO = PDEMTFS(feat,label,opts)
% Parameters
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c1'), c1 = opts.c1; end 
if isfield(opts,'c2'), c2 = opts.c2; end 
if isfield(opts,'Lasso'), Lasso_weight = opts.Lasso; end 
if isfield(opts,'weight'), Filter_weight = opts.weight; end 
%if isfield(opts,'Vmax'), Vmax = opts.Vmax; end 
%if isfield(opts,'thres'), thres = opts.thres; end
X = feat;
Y = label;

T = 4   ; % 生成T个相关任务
tasks = cell(1, T); % 初始化任务存储变量
X_task = cell(1, T); % 初始化迁移任务存储变量


% ----------------利用Lasso范数计算选择概率-------------------
% 使用原始权重（未排序）与阈值比较来划分Promising和Remaining Sets
promising_idx = find(Lasso_weight >= 1);
remaining_idx = find(Lasso_weight < 1);
% 计算 Lasso_weight > 0 的特征数量
num_promising = sum(Lasso_weight < 1);
% 计算总特征数
num_features = length(Lasso_weight);
% 计算 p_promising
p_promising = num_promising / num_features;
% 确保 p_promising 在 [0,1] 范围内
 p_promising = max(0, min(1, p_promising));
% 计算 p_remaining
p_remaining = 1 - p_promising;

% ---------------task run--------------------
for t = 1:T
    % 对于每个任务，根据概率选择特征
    selected_promising = randsample(promising_idx, round(p_promising * num_features), true);
    selected_remaining = randsample(remaining_idx, round(p_remaining * num_features), true);
    % 合并选择的特征并存储为一个任务
    tasks{t} = union(selected_promising, selected_remaining);
end

%---------------------Algorithm operation---------------------
alpha = 0.9; % 适应度函数中的权重
numFeatures = size(feat, 2); % 特征数量
numParticles = N; % 粒子数量

me = max_Iter;
c1=2.5-(1:me).*(2.0./me);
c2=0.5+(1:me).*(2.0./me);
w=0.9-(1:me).*(0.5./me);

%%%%%%
particles = cell(T, 1);
globalBests = cell(T, 1);
% Initialize particles for each task
for t = 1:T
    featureSubset = tasks{t}; % 任务t
    numFeaturesInTask = length(featureSubset); 
    % Initialize particles for task t
    for i = 1:numParticles
        % 初始化二进制
        particlePosition = false(1, numFeatures); 
        particlePosition(featureSubset) = rand(1, numFeaturesInTask) > 0.6;

        particles{t}(i).position = particlePosition;
        particles{t}(i).velocity = zeros(1, numFeatures); 
        particles{t}(i).fitness = Inf;
        particles{t}(i).pbest = particlePosition;
        particles{t}(i).pbestFitness = Inf;
    end
    % Initialize the global best for task t

    globalBests{t} = struct('position',particles{t}(1).position, 'fitness', Inf);  % Initialize each cell as a struct
end

% Initialize migration task X_task
X_task{1} = true(1,numFeatures);
X_task{2} = true(1,numFeatures);
X_task{3} = true(1,numFeatures);
X_task{4} = true(1,numFeatures);
NC = 0; % Initial value, to add features
AC = 0; % Initial value, to add features
NC_M = 0;
AC_M = 0;
NC_decrease = 0;
NC_increase = 0;
AC_decrease = 0;
AC_increase = 0;
%%%%%% Pre
curve = zeros(max_Iter,1);
curve(1) = Inf;
FeaturesCount = zeros(1,max_Iter);
FeaturesCount(1) = Inf;
maxAccuracy = 0;
maxAccuracyFeaturesCount = 0;
Time_curve = zeros(max_Iter,1);
Time_curve(1) = 0;
Mean_Acc = zeros(max_Iter,1);
Mean_Acc(1) = Inf;
Mean_Num = zeros(max_Iter,1);
Mean_Num(1) = Inf;
Acc_NP = zeros(T,N);
Num_NP = zeros(T,N);
Acc_NP(1) = 0;
Num_NP(1) = 0;
% Optimization Loop
for iter = 1:max_Iter
    % start time
    tic
    parfor t = T % 并行循环
        % 从tasks中获取当前任务的特征子集
        featureSubset = tasks{t};
        for i = 1:numParticles
            %保存该粒子位置
            currentPosition=particles{t}(i).position;
            % 更新粒子速度
            r1 = rand();
            r2 = rand();
            particles{t}(i).velocity = w(iter) * particles{t}(i).velocity + ...
                                       c1(iter) * r1 .* (particles{t}(i).pbest - particles{t}(i).position) + ...
                                       c2(iter) * r2 .* (globalBests{t}.position - particles{t}(i).position);
             % 预更新粒子位置
            newPosition = currentPosition + particles{t}(i).velocity;


            % 适用于特征选择的二值化处理，确保只选择当前任务相关的特征
            selectedFeatures = false(1, numFeatures); % Initialize with false 
            selectedFeatures(featureSubset) = particles{t}(i).position(featureSubset) > 0.6;
            % 检查是否有特征被选中
            if sum(selectedFeatures) == 0
                % 如果没有特征被选中，保持粒子在原位置，不更新粒子状态，直接跳过该粒子的后续处理
%                 fprintf('任务 %d, 粒子 %d: 没有特征被选中，保持当前状态不变。\n', t, i);
                continue;  % 跳过该粒子的后续处理
            end
            % 有特征被选中，更新粒子位置
            particles{t}(i).position = newPosition;
            particles{t}(i).fitness = calculateFitness(selectedFeatures, X, Y, alpha);
            
            Acc_NP(t,i) = particles{t}(i).fitness;
            Num_NP(t,i) = sum(particles{t}(i).position>0.6);
            % 更新任务个体最佳
            if isempty(particles{t}(i).pbest) || particles{t}(i).fitness < particles{t}(i).pbestFitness
                particles{t}(i).pbest = particles{t}(i).position;
                particles{t}(i).pbestFitness = particles{t}(i).fitness;
            end
            % 更新任务全局最优解
            if isempty(globalBests{t}) || particles{t}(i).fitness < globalBests{t}.fitness
                globalBests{t}.position = particles{t}(i).position;
                globalBests{t}.fitness = particles{t}(i).fitness;
            end
        end
    end
  
    Mean_Acc(iter) = mean(Acc_NP(:));
    Mean_Num(iter) = mean(Num_NP(:));
    if iter >1
        if  Mean_Num(iter)<Mean_Num(iter-1)
            NC = 0;
            NC_decrease = NC_decrease + 1;
        elseif Mean_Num(iter)>Mean_Num(iter-1)
            NC = 1;
            NC_increase = NC_increase + 1;
        end
        if  Mean_Acc(iter)<Mean_Acc(iter-1)
            AC = 0;
            AC_decrease = AC_decrease + 1;
        elseif Mean_Acc(iter)>Mean_Acc(iter-1)
            AC = 1;
            AC_increase = AC_increase + 1;
        end
        
        % 通过FFT更新每个任务，建立任务之间的联系
        X_task{1} = globalBests{1}.position;
        X_task{2} = globalBests{2}.position;
        X_task{3} = globalBests{3}.position;
        X_task{4} = globalBests{4}.position;
        % Dynamic multitasking strategy
        FFT = DMT(NC,AC,X_task,feat,label,opts);
        [particles{1}.position] = deal(FFT.sf1);
        [particles{2}.position] = deal(FFT.sf2);
        [particles{4}.position] = deal(FFT.sf3);
        [particles{3}.position] = deal(FFT.sf4);

        % Task migration and knowledge transfer strategy
        if NC_decrease>2 || NC_increase>2 || AC_decrease>2 || AC_increase>2
            % ---------任务知识迁移：根据特征数量和准确率的变化调整任务---------
            if NC_decrease>2
                NC_M = 0;
                NC_decrease = 0;
            end
            if NC_increase>2
                NC_M = 1;
                NC_increase = 0;
            end
            if AC_decrease>2
                AC_M = 1;
                AC_decrease = 0;
            end
            if AC_increase>2
                AC_M = 0;
                AC_increase = 0;
            end
            FFT = TMKT(NC_M,AC_M,X_task,feat,label,opts);
            [particles{1}.position] = deal(FFT.sf1);
            [particles{2}.position] = deal(FFT.sf2);
            [particles{4}.position] = deal(FFT.sf3);
            [particles{3}.position] = deal(FFT.sf4);
        end
    end
    elapsedTime = toc;
    Time_curve(iter) = elapsedTime;
    % 获取每一代的适应值
    for t = 1: T
        % 提取对应任务的最佳特征子集的索引
        selectedFeatures = globalBests{t}.position>0.6;
        % 计算选取的特征个数
        numSelectedFeatures = sum(selectedFeatures);
        % 检查是否有特征被选中
        if sum(selectedFeatures) == 0
            % 如果没有特征被选中，保持任务中的原位置，不更新任务状态，直接跳过该任务的后续处理
        %                 fprintf('任务 %d: 没有特征被选中，保持当前状态不变。\n', t);
            continue;  % 跳过该任务的后续处理
        end
        % 该特征子集的分类正确率
        accuracy = evaluateFeatureSubset(selectedFeatures, feat, label, 5);
        if accuracy > maxAccuracy
            maxAccuracy = accuracy;
            maxAccuracyFeaturesCount = numSelectedFeatures;
        end
    end
    % 储存每次迭代的收敛值
    curve(iter) = 1-maxAccuracy;
    FeaturesCount(iter) = maxAccuracyFeaturesCount;
        
end


% Store results
PSO.c  = curve;
PSO.acc = maxAccuracy;
PSO.nf = maxAccuracyFeaturesCount;
PSO.f  = feat;
PSO.l  = label;
PSO.time = sum(Time_curve);
end

%---------------------Dynamic multitasking strategy-----------------
function DMT = DMT(NC,AC,X_task,feat,label,opts)
    Filter_weight = opts.weight;
    % 初始定义
    sFeat1 = X_task{1};
    sFeat2 = X_task{2};
    sFeat3 = X_task{3};
    sFeat4 = X_task{4}; 

    % Task scenario processing
    if AC == 1 % AC increase
        sFeat1 = X_task{1} | Filter_weight.*X_task{3}>0.6;
    elseif AC ==0
        sFeat2 = X_task{1} & Filter_weight.*X_task{3}>0.6;  
    end
        
    if NC == 1 % NC increase
        sFeat3 = X_task{2} | Filter_weight.*X_task{4}>0.6;
    elseif NC == 0
        sFeat4 = X_task{2} & Filter_weight.*X_task{4}>0.6; 
    end

% Select features based on selected index
Sf1    = sFeat1 > opts.thres;
Sf2    = sFeat2 > opts.thres; 
Sf3    = sFeat3 > opts.thres; 
Sf4    = sFeat4 > opts.thres; 
sFeat1 = feat(:,Sf1);
sFeat2 = feat(:,Sf2);
sFeat3 = feat(:,Sf3);
sFeat4 = feat(:,Sf4);
% Store results
DMT.sf1 = Sf1; 
DMT.sf2 = Sf2; 
DMT.sf3 = Sf3; 
DMT.sf4 = Sf4; 
DMT.ff1 = sFeat1;
DMT.ff2 = sFeat2;
DMT.ff3 = sFeat3;
DMT.ff4 = sFeat4;
DMT.nf1 = length(Sf1);
DMT.nf2 = length(Sf2);
DMT.nf3 = length(Sf3);
DMT.nf4 = length(Sf4);
DMT.f  = feat;
DMT.l  = label;
end

%--------------Task migration and knowledge transfer strategy--------
function TMKT = TMKT(NC_M,AC_M,X_task,feat,label,opts)
    Filter_weight = opts.weight;
    % Task creation
    sFeat1 = X_task{1};
    sFeat2 = X_task{2};
    sFeat3 = X_task{3};
    sFeat4 = X_task{4}; 
    % ---------Task scenario processing---------
    if AC_M == 1 % AC increase
        sFeat1 = X_task{1} | X_task{2} | Filter_weight.*(X_task{3}>0.6);
    elseif AC_M ==0
        sFeat2 = X_task{1} & X_task{2} & Filter_weight.*(X_task{3}>0.6);  
    end
        
    if NC_M == 1 % NC increase
        sFeat3 = X_task{1} | Filter_weight.*(X_task{3}>0.6) | Filter_weight.*(X_task{4}>0.6);
    elseif NC_M == 0
        sFeat4 = X_task{2} & Filter_weight.*(X_task{3}>0.6) & Filter_weight.*(X_task{4}>0.6); 
    end

% Select features based on selected index
Sf1    = sFeat1 > opts.thres;
Sf2    = sFeat2 > opts.thres; 
Sf3    = sFeat3 > opts.thres; 
Sf4    = sFeat4 > opts.thres; 
sFeat1 = feat(:,Sf1);
sFeat2 = feat(:,Sf2);
sFeat3 = feat(:,Sf3);
sFeat4 = feat(:,Sf4);
% Store results
TMKT.sf1 = Sf1; 
TMKT.sf2 = Sf2; 
TMKT.sf3 = Sf3; 
TMKT.sf4 = Sf4; 
TMKT.ff1 = sFeat1;
TMKT.ff2 = sFeat2;
TMKT.ff3 = sFeat3;
TMKT.ff4 = sFeat4;
TMKT.nf1 = length(Sf1);
TMKT.nf2 = length(Sf2);
TMKT.nf3 = length(Sf3);
TMKT.nf4 = length(Sf4);
TMKT.f  = feat;
TMKT.l  = label;
end


