import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix'):
    """
    Plot a confusion matrix for true vs predicted labels.
    
    Parameters:
        y_true (list or array-like): True labels.
        y_pred (list or array-like): Predicted labels.
        labels (list, optional): List of label names to use in the plot.
        title (str, optional): Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Print a detailed classification report.
    
    Parameters:
        y_true (list or array-like): True labels.
        y_pred (list or array-like): Predicted labels.
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)
