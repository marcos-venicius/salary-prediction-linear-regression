import matplotlib.pyplot as plt


def plot(name: str, x, y, predictions):
    plt.title(name)
    plt.scatter(x, y, label='Real', color='green', alpha=0.3)
    plt.scatter(x, predictions, label='Predicted', color='red', alpha=0.3)
    plt.plot(x, predictions, color='blue')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()
