import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    def convert_lists_to_integers(lst):
        return np.array([int(i) for elem in lst for i in elem]).sum()

    data['X'] = data['X'].apply(convert_lists_to_integers)
    data['A1'] = data['A1'].apply(lambda x: ast.literal_eval(x))

    return data

def calc_correlations(data):
    corr_xa1 = np.corrcoef(data['X'].values.reshape(-1, 1), np.stack(data['A1']).reshape(-1, 1))[1][0]
    corr_xy = np.corrcoef(data['X'].values.reshape(-1, 1), data['Y'].values.reshape(-1, 1))[0][1]
    corr_ya1 = np.corrcoef(data['Y'].values.reshape(-1, 1), np.stack(data['A1']).reshape(-1, 1))[1][0]

    return corr_xa1, corr_xy, corr_ya1

def plot_correlations(corr_xa1, corr_xy, corr_ya1):
    plt.figure(figsize=(5, 5))

    correlation_matrix = [[1, corr_xa1, corr_xy],
                          [corr_xa1, 1, corr_ya1],
                          [corr_xy, corr_ya1, 1]]

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Heatmap of Correlations');
    plt.show()

def main():
    file_path = "/Users/aakashagarwal/Documents/GitHub/palindrome_classifier/Code/model/visualization/evaluation_results_num_1's.csv"
    data = load_data(file_path)

    corr_xa1, corr_xy, corr_ya1 = calc_correlations(data)

    print('Correlation X - A1: %.4f' % corr_xa1)
    print('Correlation X - Y: %.4f' % corr_xy)
    print('Correlation Y - A1: %.4f' % corr_ya1)

    plot_correlations(corr_xa1, corr_xy, corr_ya1)

if __name__ == "__main__":
    main()