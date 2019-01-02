import pandas as pd
import matplotlib.pyplot as plt

# reading the results csv file
results = pd.read_csv('Results_100_epochs/ResNet50/4_ResNet50_Baseline_Paper_200epochs/ResNet50_no_aug__map.csv', delimiter=';')

# Plotting the Loss function

resultsPlot = plt.figure(figsize=(14,3))


plt.subplot(1, 2, 1)
plt.plot(results['Epoch'], results['Train loss'], label = 'Train Loss')
plt.plot(results['Epoch'], results['Val loss'], label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,20)
plt.legend()
#lossPlot.savefig('Results/ResNet152/2_5degRot_200epochs/validation.png')
#plt.savefig('validation.png')



# Plotting the MAP's
plt.subplot(1, 2, 2)
plt.plot(results['Epoch'], results['Train_mAP'], label = 'Train MAP')
plt.plot(results['Epoch'], results['Val_mAP'], label = 'Validation MAP')
plt.xlabel('Epoch')
plt.ylabel('MAP')
plt.ylim(0,1)


plt.legend()

resultsPlot.suptitle('ID 7', fontsize=16)
resultsPlot.subplots_adjust(bottom=0.2)

resultsPlot.savefig('Results_100_epochs/ResNet152/4_mix_aug_dropout_bn_200epochs/MAP_VAL_ID_7.png')

plt.show()

#APs = pd.read_csv('Results.csv', delimiter=',')