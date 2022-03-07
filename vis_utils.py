import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show_images(image_list, colorbar=False):
    N = len(image_list)
    fig, axs = plt.subplots(1, N, figsize=(7 * N,7))
    for n in range(N):
        im1 = axs[n].imshow(image_list[n])
        if colorbar:
            divider = make_axes_locatable(axs[n])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')
    plt.show()
    return fig

