"""
Function to visualize the results
"""

def show_image(output):
  image_grid = vutils.make_grid(
    output,
    nrow=4,
    normalize=True,)
  image_grid_np = image_grid.cpu().numpy().transpose(1, 2, 0)*255
  image_grid_np = np.uint8(image_grid_np)
  IPython.display.display(Image.fromarray(image_grid_np))


def show_tensor_images(image_tensor, num_images=25, size=(1, 64, 64)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
