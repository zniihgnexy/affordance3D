import numpy as np

from affordance3d.mem import segmenter as G


def test_mask_boxes_keep_part_sized_segments_and_drop_duplicates() -> None:
    small, part, wall = (np.zeros((20, 20), dtype=bool) for _ in range(3))
    small[0, 0:3] = True  # 3 pixels: too small
    part[4:8, 5:10] = True  # 20 pixels
    wall[:, :] = True  # the whole image

    boxes = G.mask_boxes([small, part, part.copy(), wall], min_pixels=10, max_share=0.3)

    assert boxes == [(5, 4, 10, 8)]
