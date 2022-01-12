function ym = Dropout(y, ratio)
  [m, n] = size(y);%copy y size  
  ym     = zeros(m, n);%ym = y size with zeros
  
  num     = round(m*n*(1-ratio));%number of non-zero elements
  idx     = randperm(m*n, num);%random (1 to m*n) with num size 
  ym(idx) = 1 / (1-ratio);%random 1 / (1-ratio) non-zero elements
                          %the rest (ratio) still zeros
end
