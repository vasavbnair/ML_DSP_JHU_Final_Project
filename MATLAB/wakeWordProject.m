scriptPath = pwd;
datasetFName = fullfile(scriptPath, "wakeWordFeatures", "datasetIndex.csv");
% Only perform feature extraction if processed dataset doesn't already exist
if ~isfile(datasetFName)
    %Perform Feature Extraction
    [datasetIdx, path2Coeff] = wakeWord_FeatureExtraction(1000, 1000, 300, 200, 300, 200);
else
    userChoice = questdlg( ...
        'Processed dataset already exists. Would you like to perform feature extraction anyway?', ...
        'Existing Dataset', ...
        'Yes', 'No', 'No');
    if strcmp(userChoice, 'Yes')
        %Perform Feature Extraction
        [datasetIdx, path2Coeff] = wakeWord_FeatureExtraction(1000, 1000, 75, 75, 75, 75);
    else
        datasetIdx = table2struct(readtable(datasetFName), 'ToScalar', true);
        path2Coeff = fullfile(scriptPath, "wakeWordFeatures", "MFCCcoeff");
    end
end

%Train the model
currModel = wakeWord_KNN_Training(datasetIdx, path2Coeff);

%Perform Prediction
[predictions, accuracy] = wakeWord_KNN_Validation(datasetIdx, path2Coeff, currModel);



% Demonstrate performance
fprintf('Validation Accuracy: %.2f%%\n', accuracy);

validationLabels = datasetIdx.isWakeWord(logical(datasetIdx.isValidation));

% Shows True Negative, True Positive, False Positive, False Negative stats
figure; confusionchart(validationLabels, predictions); title("Prediction Statistics (KNN)");