%% Bag of Words Classification on Sentiment Dataset

% Clear workspace
clear; clc;

%% Load Training Data
trainData = readtable('labeledTrainData.tsv', 'FileType', 'text', 'Delimiter', '\t');

reviewsTrain = trainData.review;
labelsTrain = trainData.sentiment;

%% Load Test Data
testData = readtable('testData.tsv', 'FileType', 'text', 'Delimiter', '\t');

reviewsTest = testData.review;

%% Tokenize and preprocess
documentsTrain = tokenizedDocument(reviewsTrain);
documentsTest = tokenizedDocument(reviewsTest);

% OPTIONAL: preprocess to remove stop words and punctuation
documentsTrain = erasePunctuation(lower(documentsTrain));
documentsTest = erasePunctuation(lower(documentsTest));
documentsTrain = removeStopWords(documentsTrain);
documentsTest = removeStopWords(documentsTest);

%% Create Bag of Words model with TF-IDF weighting
bag = bagOfWords(documentsTrain);

% Optional: Remove rare words (less noise & fewer features)
bag = removeInfrequentWords(bag, 5);

% Transform train and test reviews into vectors
XTrain = tfidf(bag, documentsTrain);
XTest = tfidf(bag, documentsTest);

% NOTE: tfidf() returns sparse matrices

%% Dimensionality Reduction with Sparse SVD
% Reduce to top k components (100 recommended)
k = 100;
fprintf('Reducing dimensionality to %d components using sparse SVD...\n', k);

[U, S, V] = svds(XTrain, k);  % sparse SVD on sparse XTrain

% Reduced feature matrices
XTrainPCA = U * S;          % training set reduced
XTestPCA = XTest * V;       % project test set using V

%% Convert labels to categorical
YTrain = categorical(labelsTrain);

%% Model 1: Support Vector Machine (SVM)
fprintf('Training SVM...\n');
svmModel = fitcsvm(XTrainPCA, YTrain, 'KernelFunction', 'linear');
YPredSVM = predict(svmModel, XTrainPCA);
accSVM = mean(YPredSVM == YTrain);
fprintf('SVM Training Accuracy: %.2f%%\n', accSVM * 100);

%% Model 2: K-Nearest Neighbors (KNN)
fprintf('Training KNN...\n');
knnModel = fitcknn(XTrainPCA, YTrain, 'NumNeighbors', 5);
YPredKNN = predict(knnModel, XTrainPCA);
accKNN = mean(YPredKNN == YTrain);
fprintf('KNN Training Accuracy: %.2f%%\n', accKNN * 100);

%% Model 3: Decision Tree
fprintf('Training Decision Tree...\n');
treeModel = fitctree(XTrainPCA, YTrain);
YPredTree = predict(treeModel, XTrainPCA);
accTree = mean(YPredTree == YTrain);
fprintf('Decision Tree Training Accuracy: %.2f%%\n', accTree * 100);

%% Summary
fprintf('\nSummary of Training Accuracies:\n');
fprintf('SVM Accuracy: %.2f%%\n', accSVM * 100);
fprintf('KNN Accuracy: %.2f%%\n', accKNN * 100);
fprintf('Decision Tree Accuracy: %.2f%%\n', accTree * 100);

%% (Optional) Predict Test Data & Save Results
% For example, with SVM:
fprintf('Generating predictions on test data using SVM...\n');
YPredTestSVM = predict(svmModel, XTestPCA);

% Prepare submission table
submission = table(testData.id, double(YPredTestSVM)-1, 'VariableNames', {'id', 'sentiment'});

% Write submission to CSV
writetable(submission, 'submission.csv');
fprintf('Predictions saved to submission.csv.\n');

