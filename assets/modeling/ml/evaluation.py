from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import classification_report, roc_curve, auc
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_metrics(history):
    """Plot the loss, auc score, precision, and recall against training history.
    This function is only used for local training.
    
    Args: 
        history(object): training history from TensorFlow model. 
    Returns:
        None: Plot results
    """
    mpl.rcParams['figure.figsize'] = (12, 10)
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.5,1])
        else:
            plt.ylim([0,1])
        plt.legend()
        
        
def plot_test_auc(y_test, y_prob):
    """Plot AUC curve for testing data.
    This function is only used for SageMaker hosted training, testing phase.
    
    Args:
        y_test(numpy.array): true labels of the testing data
        y_prob(numpy.array): predicted probabilities from the output of the model
        
    Returns:
        None: Plot results
    """
    mpl.rcParams['figure.figsize'] = (5, 5)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_socre = auc(fpr, tpr)
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test AUC={:.2f}'.format(auc_socre))
    # show the plot
    plt.show()