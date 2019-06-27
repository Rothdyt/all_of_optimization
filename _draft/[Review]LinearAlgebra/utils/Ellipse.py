import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class Ellipse2d:
    def __init__(self, center, width, height, angle):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.ellipse = None

    def create(self, facecolor='white', edgecolor='tomato', linestyle='-', **kwargs):
        self.ellipse = Ellipse(xy=self.center, width=self.width,
                               height=self.height, angle=self.angle,
                               facecolor=facecolor, edgecolor=edgecolor,
                               linestyle=linestyle, **kwargs)
        return self.ellipse

    def draw(self, figsize=(6, 6), save=False):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        if self.ellipse == None:
            self.create()
        ax.add_patch(self.ellipse)
        boudary = max(self.width, self.height) * 1.2
        plt.xlim(-boudary, boudary)
        plt.ylim(-boudary, boudary)
        if save:
            plt.savefig("./ellipse.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    # e = Ellipse2d((0, 0), width=2, height=2/3, angle=45)
    # e.draw(save=True)

    params = {"Before Rotation": 0, "After Rotation": -45}
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    for ax, (title, degree) in zip(axs, params.items()):
        ax.axvline(c='grey', lw=1)
        ax.axhline(c='grey', lw=1)
        ellipse = Ellipse2d((0, 0), width=4, height=2/3, angle=degree)
        ax.add_patch(ellipse.create())
        ax.set_title(title)
        boudary = 2 * 1.2
        ax.set_xlim(-boudary, boudary)
        ax.set_ylim(-boudary, boudary)
    # plt.show()
    plt.savefig("./ellipse.png", dpi=300)
