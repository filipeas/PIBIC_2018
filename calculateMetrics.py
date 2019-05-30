import csv
from statistics import mean
import numpy as np

dataset = [
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20'
]

percentages = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5
]

results_path = 'results/'

total_means = [[[],[],[], []] for i in range(len(percentages))]

general = open(results_path + 'general.csv', 'w')
total_results = csv.writer(general, delimiter='/')
total_results.writerow(['', 'Accuracy', '','Sensitivity', '', 'Specificity', '', 'Dice', ''])
total_results.writerow(['', 'Mean', 'Std','Mean', 'Std', 'Mean', 'Std', 'Mean', 'Std'])

for image in dataset:
    csv_file = open(f'results/results_{image}.csv')
    reader = csv.reader(csv_file)

    for index,row in enumerate(list(reader)[1:]):
        for index_j, value in enumerate(row[1:]):
            total_means[index][index_j].append(float(value)*100)


for index, percent in enumerate(total_means):
    total_results.writerow([str(percentages[index]), str(mean(percent[0])), str(np.std(percent[0])),str(mean(percent[1])), str(np.std(percent[1])), str(mean(percent[2])), str(np.std(percent[2])), str(mean(percent[3])), str(np.std(percent[3]))])

general.close()
            