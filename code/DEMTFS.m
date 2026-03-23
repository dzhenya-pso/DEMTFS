% Task creation through FFT  (2/22/2025)

% == == == == == == == == Input == == == == == == == == %
% NC  : Whether the number of features changes          %
% AC  : Whether accuracy changes                        %
% feat  : Features                                      %
% label : Labels (categories)                           %
% X     : Decision vector								%
% opts  : Parameter structure for evolutionary algorithm%
% == == == == == == == == == == == == == == == == == == %

% == == == == == == == == Output == == == == == == == ==%
% task  : Processed task                                %
% == == == == == == == == == == == == == == == == == == %
function PSO = DEMTFS(feat,label,opts)
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

T = 4   ; % Generate T related tasks
tasks = cell(1, T); % Initialize task storage variable
X_task = cell(1, T); % Initialize transfer task storage variable


% ----------------Calculate selection probability using Lasso norm-------------------
% Use original weights (unsorted) compared with threshold to divide Promising and Remaining Sets
promising_idx = find(Lasso_weight >= 1);
remaining_idx = find(Lasso_weight < 1);
% Count the number of features with Lasso_weight > 0
num_promising = sum(Lasso_weight < 1);
% Calculate the total number of features
num_features = length(Lasso_weight);
% Calculate p_promising
p_promising = num_promising / num_features;
% Ensure p_promising is within the range [0,1]
 p_promising = max(0, min(1, p_promising));
% Calculate p_remaining
p_remaining = 1 - p_promising;

% ---------------% task run--------------------
for t = 1:T
    selected_promising = randsample(promising_idx, round(p_promising * num_features), true);
    selected_remaining = randsample(remaining_idx, round(p_remaining * num_features), true);
    tasks{t} = union(selected_promising, selected_remaining);
end

%---------------------Algorithm operation---------------------
alpha = 0.9; 
numFeatures = size(feat, 2); % Number of features
numParticles = N; % Population size

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
    parfor t = 1: T % Parallel loop parfor for T tasks
        % Get the feature subset of the current task from tasks
        featureSubset = tasks{t};
        for i = 1:numParticles
		
            currentPosition=particles{t}(i).position;
            % Update population
            r1 = rand();
            r2 = rand();
            particles{t}(i).velocity = w(iter) * particles{t}(i).velocity + ...
                                       c1(iter) * r1 .* (particles{t}(i).pbest - particles{t}(i).position) + ...
                                       c2(iter) * r2 .* (globalBests{t}.position - particles{t}(i).position);
            
            newPosition = currentPosition + particles{t}(i).velocity;


            % Binarization for feature selection, ensuring only current task-related features are selected
            selectedFeatures = false(1, numFeatures); % Initialize with false 
            selectedFeatures(featureSubset) = particles{t}(i).position(featureSubset) > 0.6;
            % Check if any features are selected
            if sum(selectedFeatures) == 0
                continue;  % Skip subsequent processing of this particle
            end
            % Features are selected, update particle
            particles{t}(i).position = newPosition;
            particles{t}(i).fitness = calculateFitness(selectedFeatures, X, Y, alpha);
            
            Acc_NP(t,i) = particles{t}(i).fitness;
            Num_NP(t,i) = sum(particles{t}(i).position>0.6);
            % Update the individuals for the task
            if isempty(particles{t}(i).pbest) || particles{t}(i).fitness < particles{t}(i).pbestFitness
                particles{t}(i).pbest = particles{t}(i).position;
                particles{t}(i).pbestFitness = particles{t}(i).fitness;
            end
            % Update the global for the task
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
        
        % Update each task via FFT and establish connections between tasks
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
            % ---------% Task knowledge transfer: Adjust tasks based on changes in feature number and accuracy---------
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
    % Obtain the fitness value for each generation
    for t = 1: T
        % Extract the index of the optimal feature subset for the corresponding task
        selectedFeatures = globalBests{t}.position>0.6;
        % Calculate the number of selected features
        numSelectedFeatures = sum(selectedFeatures);
        % Check if any features are selected
		if sum(selectedFeatures) == 0
			continue;  % Skip subsequent processing of this particle
		end
        % Calculate classification accuracy
        accuracy = evaluateFeatureSubset(selectedFeatures, feat, label, 5);
        if accuracy > maxAccuracy
            maxAccuracy = accuracy;
            maxAccuracyFeaturesCount = numSelectedFeatures;
        end
    end
    % Store the convergence value for each iteration
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
    % Pre
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


