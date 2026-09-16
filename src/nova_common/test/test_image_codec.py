from nova_common import image_codec


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
