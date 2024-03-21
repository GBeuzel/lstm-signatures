import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(history, filepath):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filepath)
    plt.close()

def plot_cm(model, X, y, filepath):
    y_pred = model.predict(X).argmax(axis=1)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, fmt='g',)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(filepath)
    plt.close()