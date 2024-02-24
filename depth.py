import torch
import cv2
import tempfile
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from IPython.display import clear_output

# Zoe_N
repo = "isl-org/ZoeDepth"
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)
clear_output(wait=True)


def get_depth(input_image):

    # Generate low resolution image.
    input_image = input_image.convert("RGB")
    low_res_depth = zoe.infer_pil(input_image)
    low_res_scaled_depth = (
            2 ** 16 - (low_res_depth - np.min(low_res_depth)) *
            2 ** 16 / (np.max(low_res_depth) - np.min(low_res_depth)))

    # Store filters in lists.
    im = np.asarray(input_image)

    tile_sizes = [[4, 4], [8, 8]]

    filters = []

    for tile_size in tile_sizes:

        num_x = tile_size[0]
        num_y = tile_size[1]

        m = im.shape[0] // num_x
        n = im.shape[1] // num_y

        filter_dict = {
            'right_filter': np.zeros((m, n)),
            'left_filter': np.zeros((m, n)),
            'top_filter': np.zeros((m, n)),
            'bottom_filter': np.zeros((m, n)),
            'top_right_filter': np.zeros((m, n)),
            'top_left_filter': np.zeros((m, n)),
            'bottom_right_filter': np.zeros((m, n)),
            'bottom_left_filter': np.zeros((m, n)),
            'filter': np.zeros((m, n))}

        for i in range(m):
            for j in range(n):
                x_value = 0.998 * np.cos((abs(m / 2 - i) / m) * np.pi) ** 2
                y_value = 0.998 * np.cos((abs(n / 2 - j) / n) * np.pi) ** 2

                if j > n / 2:
                    filter_dict['right_filter'][i, j] = x_value
                else:
                    filter_dict['right_filter'][i, j] = x_value * y_value

                if j < n / 2:
                    filter_dict['left_filter'][i, j] = x_value
                else:
                    filter_dict['left_filter'][i, j] = x_value * y_value

                if i < m / 2:
                    filter_dict['top_filter'][i, j] = y_value
                else:
                    filter_dict['top_filter'][i, j] = x_value * y_value

                if i > m / 2:
                    filter_dict['bottom_filter'][i, j] = y_value
                else:
                    filter_dict['bottom_filter'][i, j] = x_value * y_value

                if j > n / 2 and i < m / 2:
                    filter_dict['top_right_filter'][i, j] = 0.998
                elif j > n / 2:
                    filter_dict['top_right_filter'][i, j] = x_value
                elif i < m / 2:
                    filter_dict['top_right_filter'][i, j] = y_value
                else:
                    filter_dict['top_right_filter'][i, j] = x_value * y_value

                if j < n / 2 and i < m / 2:
                    filter_dict['top_left_filter'][i, j] = 0.998
                elif j < n / 2:
                    filter_dict['top_left_filter'][i, j] = x_value
                elif i < m / 2:
                    filter_dict['top_left_filter'][i, j] = y_value
                else:
                    filter_dict['top_left_filter'][i, j] = x_value * y_value

                if j > n / 2 and i > m / 2:
                    filter_dict['bottom_right_filter'][i, j] = 0.998
                elif j > n / 2:
                    filter_dict['bottom_right_filter'][i, j] = x_value
                elif i > m / 2:
                    filter_dict['bottom_right_filter'][i, j] = y_value
                else:
                    filter_dict['bottom_right_filter'][i, j] = x_value * y_value

                if j < n / 2 and i > m / 2:
                    filter_dict['bottom_left_filter'][i, j] = 0.998
                elif j < n / 2:
                    filter_dict['bottom_left_filter'][i, j] = x_value
                elif i > m / 2:
                    filter_dict['bottom_left_filter'][i, j] = y_value
                else:
                    filter_dict['bottom_left_filter'][i, j] = x_value * y_value

                filter_dict['filter'][i, j] = x_value * y_value

        filters.append(filter_dict)

    # Filters second section.
    compiled_tiles_list = []

    for i in range(len(filters)):

        num_x = tile_sizes[i][0]
        num_y = tile_sizes[i][1]

        m = im.shape[0] // num_x
        n = im.shape[1] // num_y

        compiled_tiles = np.zeros((im.shape[0], im.shape[1]))

        x_coords = list(range(0, im.shape[0], im.shape[0] // num_x))[:num_x]
        y_coords = list(range(0, im.shape[1], im.shape[1] // num_y))[:num_y]

        x_coords_between = list(range((im.shape[0] // num_x)//2, im.shape[0], im.shape[0] // num_x))[:num_x - 1]
        y_coords_between = list(range((im.shape[1] // num_y)//2, im.shape[1], im.shape[1] // num_y))[:num_y - 1]

        x_coords_all = x_coords + x_coords_between
        y_coords_all = y_coords + y_coords_between

        for x in x_coords_all:
            for y in y_coords_all:

                depth = zoe.infer_pil(Image.fromarray(np.uint8(im[x:x + m, y:y + n])))

                scaled_depth = 2 ** 16 - (depth - np.min(depth)) * 2 ** 16 / (np.max(depth) - np.min(depth))

                if y == min(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[i]['top_left_filter']
                elif y == min(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_left_filter']
                elif y == max(y_coords_all) and x == min(x_coords_all):
                    selected_filter = filters[i]['top_right_filter']
                elif y == max(y_coords_all) and x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_right_filter']
                elif y == min(y_coords_all):
                    selected_filter = filters[i]['left_filter']
                elif y == max(y_coords_all):
                    selected_filter = filters[i]['right_filter']
                elif x == min(x_coords_all):
                    selected_filter = filters[i]['top_filter']
                elif x == max(x_coords_all):
                    selected_filter = filters[i]['bottom_filter']
                else:
                    selected_filter = filters[i]['filter']

                compiled_tiles[x:x + m, y:y + n] += (
                        selected_filter * (np.mean(low_res_scaled_depth[x:x + m, y:y + n]) +
                                           np.std(low_res_scaled_depth[x:x + m, y:y + n]) *
                                           ((scaled_depth - np.mean(scaled_depth)) / np.std(scaled_depth))))

        compiled_tiles[compiled_tiles < 0] = 0
        compiled_tiles_list.append(compiled_tiles)

    # Combine depth maps
    grey_im = np.mean(im, axis=2)

    tiles_blur = gaussian_filter(grey_im, sigma=20)
    tiles_difference = tiles_blur - grey_im

    tiles_difference = tiles_difference / np.max(tiles_difference)

    tiles_difference = gaussian_filter(tiles_difference, sigma=40)

    tiles_difference *= 5

    tiles_difference = np.clip(tiles_difference, 0, 0.999)

    combined_result = (tiles_difference * compiled_tiles_list[1] + (1 - tiles_difference)
                       * ((compiled_tiles_list[0] + low_res_scaled_depth)/2)) / 2

    logging.info("Image ndarray datatype is: " + input_image.dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='.') as tmp_file:
        Image.fromarray(_to_16_bit(combined_result)).save(tmp_file.name)
        combined_result_path = tmp_file.name

    combined_result_preview = Image.fromarray(_to_8_bit(combined_result), mode='L')

    return combined_result_preview, combined_result_path


def _to_8_bit(input_image):
    normalized_image = cv2.normalize(input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    equalized_image = cv2.equalizeHist(normalized_image)
    return equalized_image


def _to_16_bit(input_image):
    if input_image.dtype == np.uint16:
        return input_image
    else:
        input_image = np.clip(input_image, 0, np.max(input_image))
        normalized_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
        output_image = (normalized_image * 65535).astype(np.uint16)
        return output_image
