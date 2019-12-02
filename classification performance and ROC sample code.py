# precision_recall_curve() function:
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

y_score = dtreeC.predict_proba(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores[:,1])
# Finally, you can plot precision and recall as functions of the threshold value using 
# Matplotlib

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()
# ROC curve plotting
# calculate score for roc curve using predict_proba(X_test)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
# plot roc_curve

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()