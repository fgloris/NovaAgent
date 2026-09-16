from nova_common.tool_result import split_images


def test_split_images_dict_payload():
    result = {
        "ok": True,
        "image_id": "viz_1",
        "images": {"viz_1": "data:image/jpeg;base64,AAAA"},
    }
    stripped, images = split_images(result)
    assert "images" not in stripped
    assert stripped["image_id"] == "viz_1"
    assert images == {"viz_1": "data:image/jpeg;base64,AAAA"}


def test_split_images_list_payload_and_ignores_non_data():
    result = {"images": ["data:image/png;base64,BBBB", "not-a-url", 123]}
    stripped, images = split_images(result)
    assert stripped == {}
    assert images == {"image_0": "data:image/png;base64,BBBB"}


def test_split_images_without_images_and_non_dict():
    assert split_images({"ok": True}) == ({"ok": True}, {})
    assert split_images(["plain", "list"]) == (["plain", "list"], {})
