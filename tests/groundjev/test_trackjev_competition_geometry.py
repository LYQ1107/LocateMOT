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


def test_memory_subset_aligns_indices_geometry_and_crop_evidence():
    from types import SimpleNamespace
    from PIL import Image
    from locatemot.groundjev.trackjev.native_onejev_adapter import NativeOneJevAdapter

    class CaptureRequest:
        def decide(self, request, media):
            self.request, self.media = request, media
            return SimpleNamespace(answers={
                'memory_0': SimpleNamespace(noul=.9),
                'memory_1': SimpleNamespace(noul=.1),
            }), {}

    engine = CaptureRequest()
    adapter = NativeOneJevAdapter('.', engine=engine, candidate_evidence='train_current_crops',
                                  competition_geometry='xyxy', memory_transport='local_xyxy')
    group = [{'box': box} for box in ([0,0,10,20], [100,100,110,120],
                                    [10,0,20,20], [130,100,140,120])]
    result = adapter.decide_memory(image=Image.new('RGB', (160,160)), group=group,
                                  assigned={1:'10',3:'11'}, tracks=[], timestamp=0.,
                                  specification={'task':'mot'}, competition={})
    state = engine.request.state
    assert state['assigned_identity_per_candidate'] == {'0':'10','1':'11'}
    assert state['original_candidate_indices'] == [1,3]
    assert [candidate['index'] for candidate in state['candidates']] == [0,1]
    assert state['competition_iou_rows_milli'] == ['1000,0','0,1000']
    assert state['current_candidate_crop_sheet'] == '<image:2>'
    assert len(engine.media) == 2
    assert engine.media[1]['image'].size == (640,192)
    assert result['updates'] == {1:{'noul':.9},3:{'noul':.1}}
