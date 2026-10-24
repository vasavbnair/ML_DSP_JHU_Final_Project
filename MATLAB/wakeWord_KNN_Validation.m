function [knnPredict, accuracy] = wakeWord_KNN_Validation(datasetIdx, path2Coeff, knnModel)
%WAKEWORD_MODELTRAINING Summary of this function goes here
%   Detailed explanation goes here

validationData = [];
validationLabel = [];

for i_mi = 1:length(datasetIdx.relativePath)
    %only use training data
    if datasetIdx.isValidation(i_mi) == 1
        load(fullfile(path2Coeff, datasetIdx.coeffFname{i_mi}));
        validationData = [validationData; mean(currCoeff, 1), std(currCoeff, 0, 1)];
        validationLabel = [validationLabel; datasetIdx.isWakeWord(i_mi)];
    end
end

knnPredict = predict(knnModel, validationData);
accuracy = mean(knnPredict == validationLabel) * 100;

end