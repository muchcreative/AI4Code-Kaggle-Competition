import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix

def get_labels(dataset):
  '''Gets all labels from specified dataset.'''
  grouped_labels = [label_batch for _, label_batch in dataset.as_numpy_iterator()]
  labels = np.concatenate(grouped_labels, axis=0)
  return labels

def plot_roc_curve(labels, predictions, show_plot=True):
  '''Plots ROC curve and calculates ROC AUC and best threshold.
  
  Args:
    labels: ndarray
    predictions: ndarray
    show_plot: bool that defaults 'True'.
               Specifies to show plot in output cell.

  Returns:
    roc_auc: float
    best_threshold: float
    fig: matplotlib.pyplot.figure
  '''

  false_pos_rates, true_pos_rates, _ = roc_curve(labels, predictions)
  gmeans = np.sqrt(true_pos_rates * (1 - false_pos_rates))
  best_threshold = np.max(gmeans)

  best_idx = np.argmax(gmeans)
  best_tpr = true_pos_rates[best_idx]
  best_fpr = false_pos_rates[best_idx]

  roc_auc = auc(false_pos_rates, true_pos_rates)
  fig, ax = plt.subplots(figsize=(12,8))
  plt.plot(false_pos_rates,
           true_pos_rates,
           color='darkorange',
           label=f'ROC curve (area = {roc_auc:.2f})')
  plt.scatter(best_fpr,
              best_tpr,
              marker='o',
              color='red',
              label=f'Best with Threshold {best_threshold:.2f}')
  plt.plot([0, 1], [0, 1],
           color='navy',
           linestyle='--',
           label='No Skill')
  plt.axhline(y=1, 
              color='green',
              label='Optimized', 
              linestyle='--')
  ax.set(title=f'Valid Dataset: Receiver Operating Characteristic Curve',
         xlabel='False Positive Rate',
         xlim=[0,1.01],
         ylabel='True Positive Rate',
         ylim=[0,1.01])
  plt.legend(loc="lower right")
  if not show_plot:
    plt.close()
  return roc_auc, best_threshold

def plot_confusion_matrix(labels, predicted_trades, show_plot=True):
  '''Plots confusion matrix and calculates confusion matrix parameters.
  
  Args:
    labels: ndarray
    predicted_trades: ndarray
    show_plot: bool that defaults 'True'.
               Specifies to show plot in output cell.

  Returns:
    cm: np.array with shape (2,2).
        Contains confusion matrix parameters.
        Given as [[tn, tp], [fn, fp]]
    cm_perc: np.array with shape (2,2)
        Contains confusion matrix parameters divided by total labels for each parameter
        Same format as cm
    fig: matplotlib.pyplot.figure
  '''

  tn, fp, fn, tp = confusion_matrix(labels, predicted_trades).ravel()
  cm = np.asarray([[tn, tp], [fn, fp]])
  total_trades = np.sum(cm)
  cm_perc = cm/total_trades
  tn_perc, tp_perc, fn_perc, fp_perc = cm_perc.ravel()

  cm_index = ['Positive', 'Negative']
  cm_columns = ['False', 'True']
  cm_df = pd.DataFrame(cm, columns=cm_columns, index=cm_index)

  fig, ax = plt.subplots(figsize=(12,8))
  annots = [f'False Positives\n{fp}\n{fp_perc:.2%}',
            f'True Positives\n{tp}\n{tp_perc:.2%}',
            f'False Negatives\n{fn}\n{fn_perc:.2%}',
            f'True Negatives\n{tn}\n{tn_perc:.2%}']
  annots = np.asarray(annots).reshape(2,2)
  ax = sns.heatmap(cm_df,
                   cmap='Blues',
                   annot=annots,
                   fmt='',
                   ax=ax)
  ax = ax.set(title=f'Valid Dataset: Confusion Matrix with {total_trades} Labels',
              xlabel='Actual',
              ylabel='Prediction')
  if not show_plot:
    plt.close()
  return cm, cm_perc