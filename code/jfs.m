% Wrapper Feature Selection Toolbox by Jingwei Too - 9/12/2020

function model = jfs(type,feat,label,opts)
switch type
 % Traditional
  case 'DEMTFS' ; fun = @DEMTFS;
end
model = fun(feat,label,opts); 
end



