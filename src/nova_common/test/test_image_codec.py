import numpy as np

from nova_common import image_codec


def test_jpeg_metadata_and_file_url_roundtrip(tmp_path):
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    frame[..., 1] = 180
    path = tmp_path / "history" / "1-camA.jpg"
    image_codec.save_image_with_metadata(path, frame, {"camera": "camA", "origin": "raw_camera"})
    assert image_codec.read_jpeg_metadata(path)["camera"] == "camA"
    url = image_codec.image_url("history", "1-camA.jpg")
    assert url == "file://history/1-camA.jpg"
    loaded = image_codec.load_image(url, tmp_path)
    assert loaded.shape == frame.shape
    assert np.abs(loaded.astype(int) - frame.astype(int)).mean() < 5  # JPEG 有损
    assert image_codec.resolve_image_path(url, tmp_path) == path


def test_resolve_image_path_rejects_escape(tmp_path):
    try:
        image_codec.resolve_image_path("file://../secret.jpg", tmp_path)
        assert False
    except ValueError:
        pass


def test_display_size_only_shrinks():
    assert image_codec.display_size(768, 768, 768) == (768, 768)
    assert image_codec.display_size(1536, 768, 768) == (768, 384)
    assert image_codec.display_size(400, 200, 768) == (400, 200)


def test_scale_factors():
    assert image_codec.scale_factors((200, 200), (100, 100)) == (0.5, 0.5)
    assert image_codec.scale_factors((100, 100), (200, 200)) == (2.0, 2.0)


def test_convert_points_shapes():
    src, dst = (200, 200), (100, 100)
    assert image_codec.convert_points([120.0, 88.0], src, dst) == [60.0, 44.0]
    assert image_codec.convert_points([[120.0, 88.0], [200.0, 100.0]], src, dst) == [
        [60.0, 44.0],
        [100.0, 50.0],
    ]
    assert image_codec.convert_points({"cam": [120.0, 88.0]}, src, dst) == {"cam": [60.0, 44.0]}
    assert image_codec.convert_points("untouched", src, dst) == "untouched"
