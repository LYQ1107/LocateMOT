from locatemot.groundjev.native_output_repair import repair_gam_bbox_output


def test_repair_keeps_complete_pairs_and_drops_nested_malformed_fragment():
    start = "<|object_ref_start|>"
    end = "<|object_ref_end|>"
    box_start = "<|box_start|>"
    box_end = "<|box_end|>"
    raw = (
        f"{start}cat{end}{box_start}<10><20><30><40>{box_end}, "
        f"{start}broken{end}{box_start}<1><2>"
        f"{start}dog{end}{box_start}<50><60><70><80>{box_end}, "
        f"{start}cat{end}{box_start}<10><20><30><40>{box_end}"
    )

    recovered, report = repair_gam_bbox_output(raw)

    assert recovered == [
        ("cat", (10, 20, 30, 40)),
        ("dog", (50, 60, 70, 80)),
    ]
    assert report["status"] == "RECOVERED"
    assert report["duplicate_count"] == 1
    assert report["dropped_reasons"] == {"nested_object_before_box_end": 1}
    assert report["coordinates_reordered"] is False
    assert report["missing_coordinates_filled"] is False
