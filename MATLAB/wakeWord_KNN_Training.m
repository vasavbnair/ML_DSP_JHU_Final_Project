function [knnModel] = wakeWord_KNN_Training(datasetIdx, path2Coeff)
%WAKEWORD_MODELTRAINING Summary of this function goes here
%   Detailed explanation goes here

trainingData = [];
trainingLabel = [];

for i_mi = 1:length(datasetIdx.relativePath)
    %only use training data
    if datasetIdx.isTraining(i_mi) == 1
        load(fullfile(path2Coeff, datasetIdx.coeffFname{i_mi}));
        trainingData = [trainingData; mean(currCoeff, 1), std(currCoeff, 0, 1)];
        trainingLabel = [trainingLabel; datasetIdx.isWakeWord(i_mi)];
    end
end
knnModel = fitcknn(trainingData, trainingLabel, 'NumNeighbors', 5, 'Standardize', true);
end