%% Reading the data:
data_raw = csvread("epochdata_15000.csv");
epoch = data_raw(1:end,1);
model_loss = data_raw(1:end,2);
model_accuracy = data_raw(1:end,3);


%
% PRINT CODE
%
plot(epoch, [model_accuracy, model_loss], 'o-')
title("Results vs. Epoch Iterations")
legend("Model Accuracy", "Model Loss")
xlabel("Epochs")