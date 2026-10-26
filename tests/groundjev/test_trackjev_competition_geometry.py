from locatemot.groundjev.trackjev.candidate_competition import (
    competition_rows,entity_clusters,runtime_competition_boxes,sparse_edges,
)


def test_separated_objects_do_not_become_overlapping_competitors():
    # Runtime coordinates describe two disjoint 10x20 objects, at x=100/130.
    boxes=[[100.,100.,110.,120.],[130.,100.,140.,120.]]
    corrected=runtime_competition_boxes(boxes,'xyxy')
    edges=sparse_edges(corrected)
    assert edges==[]
    rows=competition_rows(corrected,entity_clusters(corrected,edges),edges)
    assert rows['competition_iou_rows_milli']==['1000,0','0,1000']
    assert rows['competition_groups']==[[0],[1]]
    # The historical transport made both objects much larger and overlapping.
    assert sparse_edges(runtime_competition_boxes(boxes,'legacy_xywh'))
    assert boxes==[[100.,100.,110.,120.],[130.,100.,140.,120.]]
