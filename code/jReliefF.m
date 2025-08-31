function RF = jReliefF(feat,label,opts)
% Parameter
K = 5;

if isfield(opts,'K'), K = opts.K; end

% Convert format to categorical
label         = categorical(label); 
% Relief-F Algorithm
[idx, weight] = relieff(feat,label,K);
% Store results 
RF.sf = idx; 
RF.f  = feat; 
RF.l  = label;
RF.s  = weight;
end

