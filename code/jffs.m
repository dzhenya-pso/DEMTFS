% Filter Feature Selection Toolbox 

function model = jffs(type,feat,label,opts)
switch type
  case'nca'    ; fun = @jNeighborhoodComponentAnalysis;
  case'pcc'    ; fun = @jPearsonCorrelationCoefficient;
  case'tv'     ; fun = @jTermVarianceFeatureSelection; 
  case'rf'     ; fun = @jReliefF; 
  case'mi'     ; fun = @MI_TOOL;
end
model = fun(feat,label,opts); 
end


