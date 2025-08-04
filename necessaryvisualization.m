%% PLOT 1: Top 20 Most Frequent Words
figure;
bag = bagOfWords(documentsTrain);  % ensure you have full bag again
topk = topkwords(bag, 20);
barh(topk.Count, 'FaceColor', [0.2 0.6 0.8]);
yticklabels(topk.Word(end:-1:1));
xlabel('Word Frequency');
ylabel('Top Words');
title('Top 20 Most Frequent Words in Training Set');
grid on;

%% PLOT 2: Confusion Matrix for SVM
figure;
confusionchart(YTrain, YPredSVM);
title('Confusion Matrix - SVM (Training Set)');
xlabel('Predicted'); ylabel('Actual');

%% PLOT 3: Accuracy Comparison of Models (Bar Plot)
modelNames = {'SVM', 'KNN', 'Decision Tree'};
accuracies = [accSVM, accKNN, accTree] * 100;

figure;
bar(accuracies, 0.5);
set(gca, 'XTickLabel', modelNames);
ylabel('Accuracy (%)');
title('Comparison of Training Accuracies');
ylim([0 100]);
grid on;

%% PLOT 4: Word Cloud of Positive Reviews
posDocs = documentsTrain(labelsTrain == 1);
bagPos = bagOfWords(posDocs);
figure;
wordcloud(bagPos);
title('Word Cloud - Positive Sentiment Reviews');

%% PLOT 5: Word Cloud of Negative Reviews
negDocs = documentsTrain(labelsTrain == 0);
bagNeg = bagOfWords(negDocs);
figure;
wordcloud(bagNeg);
title('Word Cloud - Negative Sentiment Reviews');
