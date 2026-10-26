import unittest
import torch

from locatemot.groundjev.dense_head.hungarian_criterion import HungarianCriterion,xyxy_to_cxcywh
from locatemot.groundjev.dense_head.model import targets_on_grid,boxes_to_original
from tools.groundjev.dense_head.evaluate_dense_head import summarize


class DetectionNumerics(unittest.TestCase):
    def test_padding_geometry_roundtrip(self):
        meta={'image_size':[1280,720],'resize_and_padding':{'new_width':1173,'new_height':660,'pad_width':3,'pad_height':12}}
        original=torch.tensor([[0.,0.,1280.,720.],[51.,29.,63.,76.]])
        target=targets_on_grid(original.tolist(),meta,'cpu')
        self.assertTrue(torch.allclose(boxes_to_original(xyxy_to_cxcywh(target),meta),original,atol=.001))

    def test_no_annotation_negatives_are_disabled(self):
        logits=torch.zeros(1,3,requires_grad=True);boxes=torch.full((1,3,4),.2,requires_grad=True)
        loss=HungarianCriterion()({'logits':logits,'boxes':boxes},torch.empty(0,4),exhaustive=False)['loss']
        loss.backward();self.assertEqual(float(loss),0.);self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_duplicate_is_false_positive_not_extra_recall(self):
        row={'boxes':[[0,0,10,10]],'ignored':[],'dataset':'animaltrack','sequence':'rabbit_1','category':'rabbit'}
        p={'boxes':[[0,0,10,10],[0,0,10,10]],'scores':[.9,.8]}
        result=summarize([row],[p],.5)
        self.assertEqual(result['one_to_one_Recall50'],1.)
        self.assertEqual(result['Precision'],.5);self.assertEqual(result['duplicates'],1)
        self.assertEqual(result['AP50'],1.)


if __name__=='__main__':unittest.main()
