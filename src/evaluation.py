import matplotlib.pyplot as plt

def plot_metric(history_dict: dict, metric: str='binary_accuracy', title: str='') -> None:
    ''' Plot the metrics results for Neural Network epochs

    Plot the chosen metric for neural network epochs
    and saves the figure in the reports folder.

    Important: the chosen metric must be declared on the model fitting.

    Args:
        history_dict: dictionary with Neural Networks metric results
        metric: name of the metric to be ploted. Default: 'binary_accuracy'
        title: string with plot title
    Returns:
        None    
    '''
    train_result = history_dict[metric]
    validation_result = history_dict['val_' + metric]
    epochs = range(1, len(train_result) + 1)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_result, label='train')
    ax.plot(epochs, validation_result, label='validation')
    ax.set_title(title) 
    ax.set_xlabel('epoch')
    ax.set_xticks(epochs)
    plt.legend(loc='best')
    plt.savefig('../report/' + title + '_epoch_plot.svg')