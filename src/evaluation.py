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
    plt.savefig('report/' + metric + '_epoch_plot.svg')


def generate_results_str(results_list: list) -> str:
    """"Generates restults

    Generates model evaluation string from a input list
    with the loss, accuracy, precision, recall and AUC
    values respectively.

    Args:
        results_list: list with the model metrics values.

    Returns:
        string with the metrics names and values.
    """
    results_str = """
        loss: {0:2f} \t 
        accuracy: {1:2f} \t
        precision: {2:2f} \t
        recall: {3:2f} \t
        AUC: {4:2f}
    """.format(*results_list)

    return results_str