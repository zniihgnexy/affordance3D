"""The pure helpers of the game-style walkthrough overlay (report/week2_interaction.py)."""
import numpy as np
import pytest

from report import week2_interaction as G

STYLE = G.Style({"pull": (1, 2, 3), "push": (4, 5, 6), "turn": (7, 8, 9), "open": (0, 1, 0)},
                {"hook_pull": ("pull", "pull"), "key_press": ("press", "push"), "rotate": ("rotate", "turn")},
                ("open", "open"), {"rotate": "knob"}, 0.05, 0.45, 3, 30, 1.6, 0.12, 0.06, 0.05, 220, 70, 4, 13, 3, 2, 0)


@pytest.mark.parametrize(("task", "name"), [
    ("Open the left cabinet door of the TV stand", "left cabinet door"),
    ("Open the window located above the trash can in the kitchen", "window"),
    ("Increase the toaster's intensity", "toaster"),
    ("Turn on the ceiling light", "ceiling light"),
    ("Open the bottom oven door", "bottom oven door"),
    ("Pull out the second drawer from the top of the big white dresser", "second drawer"),
    ("Plug the device in the power supply located near the knife stand in the kitchen", "power supply"),
    ("Plug the device in one of the sockets behind the portable air conditioner in the corner", "sockets"),
    ("Turn on the TV using the remote control located on the TV stand", "remote control"),
])
def test_object_name_takes_the_noun_phrase_after_the_verb(task, name):
    assert G.object_name(task, "fallback") == name


def test_object_name_falls_back_without_an_object():
    assert G.object_name("Open", "hook pull") == "hook pull"
    assert G.object_name("", "knob") == "knob"


def test_rotation_far_from_the_axis_is_a_swing_and_near_it_a_twist():
    axis = np.array([0.0, 0.0, 1.0])
    assert G.is_swing("rot", np.array([0.5, 0.0, 1.0]), np.zeros(3), axis, 0.05)
    assert not G.is_swing("rot", np.array([0.01, 0.0, 1.0]), np.zeros(3), axis, 0.05)
    assert not G.is_swing("trans", np.array([0.5, 0.0, 1.0]), np.zeros(3), axis, 0.05)
    assert not G.is_swing("rot", np.array([0.5, 0.0, 1.0]), np.zeros(3), None, 0.05)


def test_action_names_come_from_the_label_and_a_swing_overrides_them():
    assert G.action_of("key_press", False, STYLE) == ("press", "push")
    assert G.action_of("hook_pull", True, STYLE) == ("open", "open")
    assert G.action_of("plug_in", False, STYLE) == ("plug in", "push")  # unknown label: its words, push


def test_phase_goes_out_and_back_once_per_period():
    fps, period = 12, 1.5
    assert G.phase(0, fps, period) == pytest.approx(0.0)
    assert G.phase(9, fps, period) == pytest.approx(1.0)  # half a period = 0.75 s = 9 frames
    assert G.phase(18, fps, period) == pytest.approx(0.0)


def test_direction_is_signed_against_the_surface_normal():
    normal, d = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.2, 0.9])
    assert G.toward(d, normal, outward=True) @ normal > 0
    assert G.toward(d, normal, outward=False) @ normal < 0


def test_a_push_arrow_slides_in_and_ends_at_the_contact_and_a_pull_starts_there():
    contact, into = np.zeros(3), np.array([0.0, 0.0, -1.0])
    tail, head = G.straight_arrow(contact, into, 1.0, 0.12, 0.06, inward=True)
    assert np.allclose(head, contact) and np.allclose(tail, [0, 0, 0.12])
    tail, head = G.straight_arrow(contact, into, 0.0, 0.12, 0.06, inward=True)
    assert np.allclose(head, [0, 0, 0.06])
    out = -into
    tail, head = G.straight_arrow(contact, out, 0.0, 0.12, 0.06, inward=False)
    assert np.allclose(tail, contact) and np.allclose(head, [0, 0, 0.12])
    tail, _ = G.straight_arrow(contact, out, 1.0, 0.12, 0.06, inward=False)
    assert np.allclose(tail, [0, 0, 0.06])


def test_arc_lies_on_the_circle_round_the_axis():
    centre, axis = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0])
    points = G.arc(centre, axis, np.array([1.0, 0.0, 0.5]), 0.05, 0.0, 180.0, n=7)
    offsets = points - centre
    assert np.allclose(np.linalg.norm(offsets, axis=1), 0.05)
    assert np.allclose(offsets[:, 2], 0.0)
    assert np.allclose(offsets[0], [0.05, 0, 0]) and np.allclose(offsets[-1], [-0.05, 0, 0], atol=1e-12)
    assert np.allclose(G.arc(centre, axis, axis, 0.05, 0.0, 90.0, n=3)[:, 2], 3.0)  # reference along the axis: any start


def test_a_tag_goes_clear_of_its_part_and_arrow_and_stays_in_the_image():
    part = (90.0, 90.0, 130.0, 130.0)  # the part and its arrow around the anchor (110, 110)
    box = G.place_tag((110.0, 110.0), 80, 30, [part], (540, 720))
    assert G.overlap(box, part) == 0
    corner = G.place_tag((535.0, 5.0), 80, 30, [], (540, 720))  # anchor in the top-right corner
    assert corner[0] >= 4 and corner[2] <= 536 and corner[1] >= 4 and corner[3] <= 716


def test_a_tag_avoids_an_earlier_tag():
    first = G.place_tag((200.0, 200.0), 80, 30, [], (540, 720))
    second = G.place_tag((200.0, 200.0), 80, 30, [first], (540, 720))
    assert G.overlap(first, second) == 0


def test_neighbouring_tags_with_the_same_object_and_action_merge():
    colour = (1, 2, 3)
    items = [((100.0, 100.0), "knob", "rotate", colour, (95, 95, 105, 105)),
             ((140.0, 100.0), "knob", "rotate", colour, (135, 95, 145, 105)),
             ((120.0, 100.0), "button", "press", colour, (115, 95, 125, 105)),  # other object: own tag
             ((400.0, 100.0), "knob", "rotate", colour, (395, 95, 405, 105))]  # too far: own tag
    merged = G.group_tags(items, 70)
    assert [m[1] for m in merged] == ["knob \u00d72", "button", "knob"]
    assert merged[0][0] == (120.0, 100.0) and merged[0][4] == (95, 95, 145, 105)
