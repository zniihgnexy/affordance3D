import numpy as np
import pytest

from affordance3d.frames import Frame
from affordance3d.mem import tokens as T
from tests.conftest import TEST_HW, TEST_K, frame_from_depth

UP = np.array([0.0, -1.0, 0.0])  # image rows grow downwards, so "up" is -y in the camera frame


def test_action_map_keeps_the_action_and_drops_the_part_identity() -> None:
    part_labels = np.array([[-1, 0], [1, 2]])

    actions = T.action_map(part_labels, part_actions=[3, 3, 5], fill_px=1)

    assert actions.tolist() == [[T.NO_ACTION, 3], [3, 5]]


def test_action_map_fills_the_gaps_between_projected_points_of_one_action() -> None:
    part_labels = np.full((9, 9), -1)
    part_labels[4, 2:7:2] = 0  # three points of one part, one pixel apart
    part_labels[0, 8] = 1  # another action, far away

    dense = T.action_map(part_labels, part_actions=[2, 6], fill_px=3)

    assert dense[4, 2:7].tolist() == [2, 2, 2, 2, 2]  # the gaps are closed
    assert (dense == 2).sum() == 5 and dense[0, 8] == 6  # nothing grows outwards, other actions are kept


def test_find_regions_splits_separate_patches_of_one_action_and_skips_small_ones() -> None:
    actions = np.full(TEST_HW, T.NO_ACTION, dtype=np.int8)
    actions[2:6, 2:6] = 1  # 16 pixels
    actions[2:6, 20:24] = 1  # same action, not touching: a second region
    actions[30, 30] = 4  # one pixel: below the minimum
    depth = np.full(TEST_HW, 2.0, dtype=np.float32)

    components, regions = T.find_regions(actions, depth, keyframe=7, min_pixels=10)

    assert [(r.action, r.pixels, r.keyframe) for r in regions] == [(1, 16, 7), (1, 16, 7)]
    assert regions[0].box == (2, 2, 6, 6) and regions[1].box == (20, 2, 24, 6)
    assert regions[1].pixel in {(21, 3), (22, 3), (21, 4), (22, 4)}  # (u, v) = (column, row), near the patch centre
    assert set(np.unique(components)) == {T.NO_REGION, 1, 2}
    assert (components == regions[1].component).sum() == 16
    assert regions[0].depth_m == pytest.approx(2.0)


def test_find_regions_drops_a_patch_without_depth() -> None:
    actions = np.full(TEST_HW, T.NO_ACTION, dtype=np.int8)
    actions[2:6, 2:6] = 0
    depth = np.zeros(TEST_HW, dtype=np.float32)

    components, regions = T.find_regions(actions, depth, keyframe=0)

    assert regions == () and not components.any()


def test_region_position_uses_only_the_regions_own_keyframe() -> None:
    region = T.Region(keyframe=0, component=1, action=0, box=(0, 0, 1, 1),
                      pixel=(int(TEST_K[0, 2]), int(TEST_K[1, 2])), pixels=12, depth_m=2.0)
    moved = np.eye(4)
    moved[:3, 3] = [1.0, 0.0, 0.0]
    frame = frame_from_depth(np.full(TEST_HW, 9.0), cam_to_world=moved)  # the depth image is not consulted

    position = T.region_position(region, frame)

    ray = np.linalg.inv(TEST_K) @ np.array([region.pixel[0], region.pixel[1], 1.0])
    assert position == pytest.approx(ray * 2.0 + [1.0, 0.0, 0.0])


def test_pool_patches_weights_each_patch_by_its_covered_area() -> None:
    grid = np.zeros((2, 2, 3), dtype=np.float32)
    grid[0, 0] = [1.0, 0.0, 0.0]
    grid[0, 1] = [0.0, 1.0, 0.0]
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True  # all of patch (0, 0)
    mask[0:4, 4:6] = True  # half of patch (0, 1)

    pooled = T.pool_patches(grid, mask)

    assert pooled == pytest.approx(np.array([2.0, 1.0, 0.0]) / np.sqrt(5.0), abs=1e-6)


def test_pool_patches_rejects_an_empty_mask() -> None:
    with pytest.raises(ValueError):
        T.pool_patches(np.ones((2, 2, 3), dtype=np.float32), np.zeros((8, 8), dtype=bool))


def test_context_box_is_square_wider_than_the_region_and_inside_the_image() -> None:
    box = T.context_box((10, 10, 20, 14), image_hw=(100, 200), scale=3.0, min_side=8)
    at_corner = T.context_box((0, 0, 4, 4), image_hw=(100, 200), scale=3.0, min_side=40)

    assert box == (0, 0, 30, 30)  # 3 x the longer side (10), shifted to stay inside
    assert at_corner == (0, 0, 40, 40)


def test_link_tokens_joins_what_looks_alike_nearby_and_separates_the_rest() -> None:
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    visual = np.array([a, b, a, a, a])
    positions = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.05, 0, 0], [2.0, 0, 0], [0.0, 0, 0]])
    actions = [0, 0, 0, 0, 1]
    keyframes = [0, 0, 1, 2, 3]

    part_of = T.link_tokens(visual, positions, actions, keyframes, min_similarity=0.5, max_distance_m=0.3)

    # token 2 joins token 0; token 1 looks different, token 3 is far away, token 4 is another action
    assert part_of.tolist() == [0, 1, 0, 2, 3]


def test_link_tokens_prefers_the_nearest_of_two_parts_that_look_the_same() -> None:
    same = np.tile([1.0, 0.0], (3, 1))
    positions = np.array([[0.0, 0, 0], [0.10, 0, 0], [0.08, 0, 0]])

    part_of = T.link_tokens(same, positions, actions=[0, 0, 0], keyframes=[0, 0, 1])

    assert part_of.tolist() == [0, 1, 1]


def test_link_tokens_needs_one_member_that_is_both_alike_and_near() -> None:
    a, b, c = np.array([1.0, 0.0]), np.array([0.6, 0.8]), np.array([0.0, 1.0])
    visual = np.array([a, b, c, a])  # the last token looks like the first, which is 0.6 m away
    positions = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]])

    part_of = T.link_tokens(visual, positions, actions=[0, 0, 0, 0], keyframes=[0, 1, 2, 3],
                            min_similarity=0.55, max_distance_m=0.25)

    # tokens 0-2 chain into one part; token 3 is near token 2 (which looks different) and alike token 0 (too far)
    assert part_of.tolist() == [0, 0, 0, 1]


def test_link_tokens_never_puts_two_regions_of_one_keyframe_in_the_same_part() -> None:
    same = np.array([[1.0, 0.0], [1.0, 0.0]])

    part_of = T.link_tokens(same, np.zeros((2, 3)), actions=[0, 0], keyframes=[4, 4])

    assert part_of.tolist() == [0, 1]


PART_ROWS, PART_COLS = slice(8, 12), slice(40, 48)  # off-centre and not square, so that swapping u and v shows
PART_PIXEL = (43, 9)  # (u, v) of the part pixel closest to its centre (43.5, 9.5)


def _wall_with_part(rows: slice = PART_ROWS) -> tuple[Frame, np.ndarray]:
    """A wall 2 m in front of the camera with a small part that sticks out 4 cm."""
    depth = np.full(TEST_HW, 2.0)
    actions = np.full(TEST_HW, T.NO_ACTION, dtype=np.int8)
    depth[rows, PART_COLS] = 1.96
    actions[rows, PART_COLS] = T.ACTIONS.index("tip_push")
    return frame_from_depth(depth), actions


def _embed_first_pixel(crops: list[np.ndarray]) -> np.ndarray:
    """Stand-in for the image-text encoder: a unit vector that depends on how many crops it was given."""
    return np.tile([1.0, 0.0], (len(crops), 1))


def test_write_memory_pools_the_patches_under_the_region_and_links_the_same_region_seen_twice() -> None:
    first, actions = _wall_with_part()
    larger, larger_actions = _wall_with_part(rows=slice(8, 14))  # the same part, seen with more pixels
    grid = np.zeros((6, 8, 4), dtype=np.float32)  # 8 x 8 pixel patches; the part lies in patch row 1, column 5
    grid[..., 0] = 1.0
    grid[1, 5] = [0.0, 0.0, 1.0, 0.0]
    seen_crops: list[tuple[int, ...]] = []

    def embed(crops: list[np.ndarray]) -> np.ndarray:
        seen_crops.extend(crop.shape for crop in crops)
        return _embed_first_pixel(crops)

    memory = T.write_memory(["10.0", "12.5"], [first, larger], [actions, larger_actions], [grid, grid], embed, up=UP)

    assert len(memory.tokens) == 2 and memory.n_parts == 1
    assert [t.time_s for t in memory.tokens] == [0.0, 2.5]
    assert memory.tokens[0].region.pixel == PART_PIXEL and memory.tokens[0].region.depth_m == pytest.approx(1.96)
    assert memory.tokens[0].visual == pytest.approx([0.0, 0.0, 1.0, 0.0])  # only the patch under the region counts
    assert seen_crops == [(TEST_HW[0], TEST_HW[0], 3)] * 2  # square context crops, clipped to the image height
    assert memory.evidence(0) == 1  # the keyframe that shows the part with more pixels
    assert memory.feature_bytes() == 2 * (4 + 2) * 2


def test_write_memory_counts_time_from_the_given_start() -> None:
    frame, actions = _wall_with_part()

    memory = T.write_memory(["10.0"], [frame], [actions], [np.ones((6, 8, 4), dtype=np.float32)], _embed_first_pixel,
                            up=UP, start_s=4.0)

    assert memory.tokens[0].time_s == pytest.approx(6.0)


def test_write_memory_without_any_region_gives_an_empty_memory() -> None:
    frame = frame_from_depth(np.full(TEST_HW, 2.0))
    nothing = np.full(TEST_HW, T.NO_ACTION, dtype=np.int8)

    memory = T.write_memory(["0.0"], [frame], [nothing], [np.ones((6, 8, 4), dtype=np.float32)], _embed_first_pixel, up=UP)

    assert memory.tokens == () and memory.n_parts == 0 and memory.feature_bytes() == 0
    with pytest.raises(ValueError):
        T.write_memory([], [], [], [], _embed_first_pixel, up=UP)


def test_read_geometry_gives_contact_normal_and_stand_point_from_the_evidence_keyframe() -> None:
    frame, actions = _wall_with_part()
    memory = T.write_memory(["0.0"], [frame], [actions], [np.ones((6, 8, 4), dtype=np.float32)], _embed_first_pixel, up=UP)

    readout = T.read_geometry(memory, token=0, frames=[frame], up=UP)

    assert readout is not None and readout.pixel == PART_PIXEL
    ray = np.linalg.inv(TEST_K) @ np.array([PART_PIXEL[0], PART_PIXEL[1], 1.0])
    assert readout.contact == pytest.approx(ray * 1.96, abs=1e-3)  # on the part, 4 cm in front of the wall
    assert abs(readout.normal @ np.array([0.0, 0.0, -1.0])) > 0.99  # the wall faces the camera
    assert readout.stand.facing @ np.array([0.0, 0.0, 1.0]) > 0.99  # the person faces the wall
    assert np.linalg.norm((readout.stand.position - readout.contact)[[0, 2]]) == pytest.approx(T.STAND_DISTANCE_M, abs=0.02)
    with pytest.raises(IndexError):
        T.read_geometry(memory, token=-1, frames=[frame], up=UP)
