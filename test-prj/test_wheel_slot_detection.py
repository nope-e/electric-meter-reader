import unittest
import numpy as np
from detect_wheel_slots import ordered_slot_crops
from recognize_wheel_slots import format_reading


class SlotDetectionTests(unittest.TestCase):
    def test_variable_count_keeps_all_boxes_and_order(self):
        rgb=np.zeros((70,420,3),np.uint8)
        for count in (5,6):
            detections=[{'xyxy':[i*60,5,i*60+35,65],'score':.7} for i in range(count)]
            slots=ordered_slot_crops(rgb,list(reversed(detections)))
            self.assertEqual(len(slots),count)
            self.assertEqual([s['xyxy'][0] for s in slots],[i*60 for i in range(count)])

    def test_missing_wheel_is_not_filled(self):
        detections=[{'xyxy':[i*60,5,i*60+35,65],'score':.7} for i in (0,1,3,4)]
        self.assertEqual(len(ordered_slot_crops(np.zeros((70,420,3),np.uint8),detections)),4)

    def test_bounds_and_crop_pixels(self):
        rgb=np.arange(60*70*3,dtype=np.uint8).reshape(60,70,3)
        slots=ordered_slot_crops(rgb,[{'xyxy':[-2.3,1.5,100,61.1],'score':.6}])
        np.testing.assert_array_equal(slots[0]['pixels'],rgb[1:60,:70])
        self.assertEqual(slots[0]['crop_xyxy'],[0,1,70,60])

    def test_decimal_format_keeps_leading_zero(self):
        self.assertEqual(format_reading('01792'),'0179.2')
        self.assertEqual(format_reading('003733'),'00373.3')
        self.assertEqual(format_reading('003733',0),'003733')

    def test_crop_margin_recovers_surrounding_pixels_without_moving_detection(self):
        rgb=np.arange(100*100*3,dtype=np.uint8).reshape(100,100,3)
        slots=ordered_slot_crops(rgb,[{'xyxy':[30,20,50,70],'score':.8}],.2,.1)
        self.assertEqual(slots[0]['xyxy'],[30,20,50,70])
        self.assertEqual(slots[0]['crop_xyxy'],[26,15,54,75])
        np.testing.assert_array_equal(slots[0]['pixels'],rgb[15:75,26:54])


if __name__=='__main__': unittest.main()
