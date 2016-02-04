function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
% cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% cost
z = theta * data;
ez = exp(z);
ezsum = sum(ez,1);

rows = labels;
cols = (1:numCases)';
idx = sub2ind(size(z), rows, cols);
ezj = ez(idx)';

J_xy = log(ezj ./ ezsum);
cost = -sum(J_xy,2) / numCases + lambda/2*sum(theta(:).^2);


% gradient
py = ez ./ repmat(ezsum,[numClasses, 1]);
diff_gr = groundTruth - py;
thetagrad = diff_gr * data' / (-numCases) + lambda * theta;





% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

