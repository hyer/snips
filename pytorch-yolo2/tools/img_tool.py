import Image
from matplotlib import pyplot as plt
if __name__ == "__main__":
    img = Image.open("/home/hyer/workspace/algo/Detection/yolo/pytorch-yolo2/predictions.jpg")
    plt.imshow(img)
    x = 200
    y =168
    plt.plot([x], [y], 'r*')
    plt.show()

