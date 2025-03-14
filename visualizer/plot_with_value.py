import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def value_above_line(f, x, y, i=-1, max_size=[0, 0], linewidth=2, line_color='red', text_color='white', box_color='hotpink', label=''):
    """
        - Hàm vẽ đường line có trực quan hóa bằng giá trị chỉ định
        Args:
            f: hàm figure của matplotlib
            x: giá trị trên trục x
            y: giá trị trên trục y
            i: vị trí cần trực quan hóa trên line
            max_size: kích thước giới hạn của text bouding box
            linewidth: độ lớn của line
    """
    if np.any(x) and np.any(y):
        max_height, max_width = max_size
        plt.plot(x, y, 
                 linewidth=linewidth, 
                 color=line_color, 
                 label=label)

        if round(np.max(y), 3) > 0.:
            temp_text = plt.text(0, 0, 
                                 f'{y[i]:0.3f}', 
                                 alpha=0,
                                 fontsize=8, 
                                 fontweight=600,
                                 color='white')
            r = f.canvas.get_renderer()
            bb = temp_text.get_window_extent(renderer=r)
            width = bb.width
            height = bb.height
            
            plt.text(x[i] + (width * 0.00027 + 0.01) * max_width, 
                    y[i] + (height * 0.0017 + 0.012) * max_height, 
                    f'{y[i]:0.3f}', 
                    fontsize=8, 
                    fontweight=600,
                    color=text_color)
            plt.gca().add_patch(
                plt.Rectangle(
                    (x[i] + width * 0.00027 * max_width, y[i] + height * 0.0017 * max_height),
                    width * 0.003 * max_width,
                    height * 0.005 * max_height,
                    # alpha=0.85,
                    facecolor=box_color
            ))
            plt.scatter(x[i], y[i], s=80, facecolor='red')
