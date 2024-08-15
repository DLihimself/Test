import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def plot_feature_correlation():
    load_path = 'store/data/train_data.csv'
    df = pd.read_csv(load_path)

    FEATURE_COL = [
      'signalStrength',
      'distance',
      'frequency',
      'temperature',
      'humidity',
      'fiberType',
      'encoding',
      'wavelength']
    data_filtered = df.reindex(columns=FEATURE_COL+['is_fault'])
    label_encoder = LabelEncoder()
    str_cols = ['fiberType', 'encoding']
    for col in str_cols:
      data_filtered[col] = label_encoder.fit_transform(data_filtered[col])
    
    corr = data_filtered.corr()
    sns.heatmap(corr, cmap='Blues', annot=True, fmt=".2f")
    plt.savefig('figs/fc_heatmap_with_label.png')

    FEATURE_COL = [
      'signalStrength',
      'distance',
      'frequency',
      'temperature',
      'humidity',
      'wavelength']
    data_filtered = df.reindex(columns=FEATURE_COL)
    corr = data_filtered.corr()
    sns.heatmap(corr, cmap='Blues', annot=True, fmt=".2f")
    plt.savefig('figs/fc_heatmap_without_label.png')

    sns.scatterplot(x=df.frequency,y=df.wavelength)
    plt.savefig('figs/frequency_wavelength_fc.png')

def plot_feature_correlation_log():
    load_path = 'store/data/train_data.csv'
    df = pd.read_csv(load_path)

    plt.loglog(df.frequency+0.01,df.wavelength+0.01)
    plt.xlabel('log(frequency)')
    plt.ylabel('log(wavelength)')
    plt.savefig('figs/frequency_wavelength_fc_log.png')


def opt_plot(step):
    data_dir = 'logs/opt' + str(step)
    train_results = os.path.join(data_dir, 'detect_model_train_results.csv')
    test_results = os.path.join(data_dir, 'detect_model_test_results.csv')
    df_train = pd.read_csv(train_results)
    df_test = pd.read_csv(test_results)
    plt.plot(df_train.epochs, df_train.acc*100, color='blue', linestyle='-', marker='o', label='train')
    plt.plot(df_test.epochs, df_test.acc*100, color='red', linestyle='--', marker='o', label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy(%)')
    out_file = f'figs/step{step}_opt_acc.png'
    plt.savefig(out_file)

def cmp_acc_plot(pre_step, cur_step):
    train_data_file = 'detect_model_train_results.csv'
    test_data_file = 'detect_model_test_results.csv'

    pre_data_dir = 'logs/opt' + str(pre_step)
    pre_train_results = os.path.join(pre_data_dir, train_data_file)
    pre_test_results = os.path.join(pre_data_dir, test_data_file)

    cur_data_dir = 'logs/opt' + str(cur_step)
    cur_train_results = os.path.join(cur_data_dir, train_data_file)
    cur_test_results = os.path.join(cur_data_dir, test_data_file)

    df_pre_train = pd.read_csv(pre_train_results)
    df_pre_test = pd.read_csv(pre_test_results)

    df_cur_train = pd.read_csv(cur_train_results)
    df_cur_test = pd.read_csv(cur_test_results)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df_pre_train.epochs, df_pre_train.acc*100, color='c', linestyle='-', marker='o', label='before')
    plt.plot(df_cur_train.epochs, df_cur_train.acc*100, color='b', linestyle='-', marker='o', label='after')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train accuracy(%)')
    plt.title('Train Set')
    #plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(df_pre_test.epochs, df_pre_test.acc*100, color='c', linestyle='--', marker='o', label='before')
    plt.plot(df_cur_test.epochs, df_cur_test.acc*100, color='b', linestyle='--', marker='o', label='after')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('test accuracy(%)')
    plt.title('Test Set')
    plt.tight_layout()

    out_file = f'figs/compare_step{pre_step}_to_step{cur_step}_opt_acc.png'
    plt.savefig(out_file)

def cmp_loss_plot(pre_step, cur_step):
    train_data_file = 'detect_model_train_results.csv'
    test_data_file = 'detect_model_test_results.csv'

    pre_data_dir = 'logs/opt' + str(pre_step)
    pre_train_results = os.path.join(pre_data_dir, train_data_file)
    pre_test_results = os.path.join(pre_data_dir, test_data_file)

    cur_data_dir = 'logs/opt' + str(cur_step)
    cur_train_results = os.path.join(cur_data_dir, train_data_file)
    cur_test_results = os.path.join(cur_data_dir, test_data_file)

    df_pre_train = pd.read_csv(pre_train_results)
    df_pre_test = pd.read_csv(pre_test_results)

    df_cur_train = pd.read_csv(cur_train_results)
    df_cur_test = pd.read_csv(cur_test_results)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df_pre_train.epochs, df_pre_train.loss, color='c', linestyle='-', marker='o', label='before')
    plt.plot(df_cur_train.epochs, df_cur_train.loss, color='b', linestyle='-', marker='o', label='after')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('Train Set')

    plt.subplot(1, 2, 2)
    plt.plot(df_pre_test.epochs, df_pre_test.loss, color='c', linestyle='--', marker='o', label='before')
    plt.plot(df_cur_test.epochs, df_cur_test.loss, color='b', linestyle='--', marker='o', label='after')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('test loss')
    plt.title('Test Set')

    plt.tight_layout()

    out_file = f'figs/compare_step{pre_step}_to_step{cur_step}_opt_loss.png'
    plt.savefig(out_file)

def plot_test_acc_for_diff_channel():
  df =pd.DataFrame(columns=['channel', 'Test_ACC'])
  df['channel'] = [8,16,24,32,40,48,56,64]
  df['Test_ACC'] = [96.98, 98.27, 98.12, 97.06, 98.39, 97.12, 97.08, 97.99]

  plt.plot(df['channel'], df['Test_ACC'], marker='o')
  plt.xticks(df['channel'])
  plt.xlabel('Numer of Channels')
  plt.ylabel('Test Accuracy')
  plt.yticks(range(90,100))
  plt.savefig('figs/diff_channels_test_acc.png')

def cmp_borplot(pre_step, cur_step):
    train_data_file = 'detect_model_train_results.csv'
    test_data_file = 'detect_model_test_results.csv'

    pre_data_dir = 'logs/opt' + str(pre_step)
    pre_train_results = os.path.join(pre_data_dir, train_data_file)
    pre_test_results = os.path.join(pre_data_dir, test_data_file)

    cur_data_dir = 'logs/opt' + str(cur_step)
    cur_train_results = os.path.join(cur_data_dir, train_data_file)
    cur_test_results = os.path.join(cur_data_dir, test_data_file)

    df_pre_train = pd.read_csv(pre_train_results)
    df_pre_test = pd.read_csv(pre_test_results)
    df_pre_train['group'] = 'before'
    df_pre_test['group'] = 'before'

    df_cur_train = pd.read_csv(cur_train_results)
    df_cur_test = pd.read_csv(cur_test_results)
    df_cur_train['group'] = 'after'
    df_cur_test['group'] = 'after'

    df_cmp = pd.DataFrame(columns=['Accuracy', 'loss', 'dataset', 'group'])
    df_cmp.loc[0] = [df_pre_train.iloc[-1]['acc']*100, df_pre_train.iloc[-1]['loss'], 'train', 'before']
    df_cmp.loc[1] = [df_pre_test.iloc[-1]['acc']*100, df_pre_test.iloc[-1]['loss'], 'test', 'before']
    df_cmp.loc[2] = [df_cur_train.iloc[-1]['acc']*100, df_cur_train.iloc[-1]['loss'], 'train', 'after']
    df_cmp.loc[3] = [df_cur_test.iloc[-1]['acc']*100, df_cur_test.iloc[-1]['loss'], 'test', 'after']

    sns.barplot(x='dataset', y='Accuracy', hue='group', data=df_cmp, palette='deep')
    out_file = f'figs/compare_step{pre_step}_to_step{cur_step}_barplot.png'
    plt.savefig(out_file)

def confusion_matrix_plot():
    cm0_file = 'logs/opt0/cm.npy'
    cm0 = np.load(cm0_file)
    cmper0 = cm0.astype('float') / cm0.sum(axis=1)[:, np.newaxis] * 100

    cm3_file = 'logs/opt3/cm.npy'
    cm3 = np.load(cm3_file)
    cmper3 = cm3.astype('float') / cm3.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(15,7))

    plt.subplot(1, 2, 1)
    sns.heatmap(cmper0, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.title('Confusion Matrix before optimization')

    plt.subplot(1, 2, 2)
    sns.heatmap(cmper3, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.title('Confusion Matrix after optimization')

    out_file = 'figs/compare_confusion_matirxs.png'
    plt.savefig(out_file)

def main():
  # feature correlation
  plot_feature_correlation()
  #plot_feature_correlation_log()

  # train and test acc for each optimization step
  opt_plot(0)
  opt_plot(1)
  opt_plot(2)
  opt_plot(3)

  # compare acc between two opt steps
  cmp_acc_plot(2,3)
  cmp_acc_plot(0,3)

  # compare loss between two opt steps
  cmp_loss_plot(2,3)
  cmp_loss_plot(0,3)

  # compare test acc for diff channels
  plot_test_acc_for_diff_channel()

  # barplot for step0 and step3
  cmp_borplot(0,3)

  # comfusion matrix plot
  confusion_matrix_plot()

if __name__ == '__main__':
    main()












