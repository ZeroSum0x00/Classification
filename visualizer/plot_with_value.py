import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt



def value_above_line(f, x, y, i=-1, max_size=[0, 0], linewidth=2,
                     line_color="red", text_color="white", box_color="hotpink", label=""):
    """
    - Hàm vẽ đường line có trực quan hóa bằng giá trị chỉ định
    Args:
        f: hàm figure của matplotlib
        x: giá trị trên trục x
        y: giá trị trên trục y
        i: vị trí cần trực quan hóa trên line
        max_size: kích thước giới hạn của text bounding box
        linewidth: độ lớn của line
    """
    if np.any(x) and np.any(y):
        max_height, max_width = max_size
        plt.plot(x, y, linewidth=linewidth, color=line_color, label=label)

        if round(np.max(y), 3) > 0.:
            value_str = f"{y[i]:0.3f}"

            # Vẽ text tạm để tính bounding box
            temp_text = plt.text(0, 0, value_str, alpha=0, fontsize=8, fontweight=600, color="white")
            r = f.canvas.get_renderer()
            bb = temp_text.get_window_extent(renderer=r)
            width = bb.width / f.get_dpi()  # Đổi pixel thành inch
            height = bb.height / f.get_dpi()

            box_width = width * max_width * 0.3
            box_x = x[i] - box_width / 2
            if max_height < 0.5:
                box_ratio = 0.8
            if max_height >= 0.5 and max_height < 1.2:
                box_ratio = 0.6
            else:
                box_ratio = 0.3
                
            space_ratio = 0.02
                
            box_height = height * max_height * box_ratio
            box_y = y[i] + space_ratio * max_height

            # Vẽ nền chữ nhật (box)
            plt.gca().add_patch(
                plt.Rectangle(
                    (box_x, box_y),  # Tọa độ box
                    box_width,       # Chiều rộng
                    box_height,      # Chiều cao
                    facecolor=box_color,
                    edgecolor="none",
                    alpha=0.7
                )
            )

            # Vẽ text **giữa box**
            plt.text(
                x[i], box_y + box_height / 2, value_str,
                fontsize=8, fontweight=600, color=text_color,
                ha="center", va="center"  # Căn giữa chữ
            )

            # Vẽ điểm tròn tại tọa độ x[i], y[i]
            plt.scatter(x[i], y[i], s=50, facecolor="red", edgecolor="black")
