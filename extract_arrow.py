from PIL import Image, ImageOps
import numpy as np

img = Image.open("arrow_images\\vertical_neg\\box_vertical_neg_300_0.png").convert("RGBA")
data = np.array(img)

r, g, b, a = data.T

white_mask = (r > 220) & (g > 220) & (b > 220)

black_mask = (r < 30) & (g < 30) & (b < 30)

arrow_mask = white_mask | black_mask

data[..., 3][~arrow_mask.T] = 0  

arrow_only = Image.fromarray(data)

arrow_only.save("arrow_transparent.png")

arrow_only.show()