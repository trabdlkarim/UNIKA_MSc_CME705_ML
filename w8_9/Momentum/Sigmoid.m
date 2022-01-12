%the pure definition of the sigmoid function
function y = Sigmoid(x)
% the element-wise division ./ 
%to account for the vector
  y = 1 ./ (1 + exp(-x));
end