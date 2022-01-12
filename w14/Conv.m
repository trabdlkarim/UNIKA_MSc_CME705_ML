%take the input image x and the convolution filter matrix W
%and return the feature maps.

function y = Conv(x, W)
%
[wrow, wcol, numFilters] = size(W);
[xrow, xcol, ~         ] = size(x);

yrow = xrow - wrow + 1;
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);

for k = 1:numFilters
  filter = W(:, :, k); 
  filter = rot90(squeeze(filter), 2);
%perform the convolution operation using conv2, a built-in
%two-dimensional convolution function of MATLAB
  y(:, :, k) = conv2(x, filter, 'valid');
end

end

