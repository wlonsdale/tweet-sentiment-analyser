% 3.2
% Import CSV file
ID = readtable("text_emotion_data_filtered.csv");

% Convert table to array
features = table2array(ID(:,"Content"));
% Tokenise the tweets
documents = tokenizedDocument(features);
% Create bag of words
bag = bagOfWords(documents);

% Remove stopwords, punctuation, numbers, and infrequent words
punctuation = ["," "?" "." ";" ":" "'" "!" "(" ")" "*" "-" "/" "\" "&"];
numbers = ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"];
newBag = removeWords(bag,[stopWords,punctuation,numbers]);
newBag = removeInfrequentWords(newBag,100);

% Create tf-idf matrix
M1 = tfidf(newBag); 
M1 = full(M1);

% Create sentiment labels
labels = table2array(ID(:,"sentiment"));
% Create labels for training
trainingLabels = labels(1:6432,:);
% Create labels for testing
testingLabels = labels(6433:end,:);

% 3.3
% Create training an testing matrixes
trainingFeatures = M1(1:6432,:);
testingFeatures = M1(6433:end,:);

% 3.4
% Training the models
% KNN - accuracy: 36%
knnModel = fitcknn(trainingFeatures,trainingLabels);
knnPrediction = predict(knnModel,testingFeatures);
confusionchart(knnPrediction,testingLabels);

% Decision Tree - accuracy: 44%
dTreeModel = fitctree(trainingFeatures,trainingLabels);
dTreePrediction = predict(dTreeModel,testingFeatures);
%confusionchart(dTreePrediction,testingLabels);

% Naive Bayes - accuracy: 31%
nbModel = fitcnb(trainingFeatures,trainingLabels);
nbPrediction = predict(nbModel,testingFeatures);
%confusionchart(nbPrediction,testingLabels);