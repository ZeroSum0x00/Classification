import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def plot_roc_curve(fpr, tpr, roc, iter):
    fig = plt.figure()
    linewidth = 2
    plt.plot(fpr, tpr, color="darkorange", linewidth=linewidth, label="ROC curve (area = %0.2f)" % roc)
    plt.plot([0, 1], [0, 1], color="navy", linewidth=linewidth, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve in epoch {iter}")
    plt.legend(loc="lower right")

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    img = img.reshape((height, width, 4))  # Chuyển thành mảng (H, W, 4)
    
    plt.close(fig)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img
