import unittest

from evaluate_local_meters import summarize


class LocalEvaluationTests(unittest.TestCase):
    def test_missing_area_is_a_failure_and_unknown_is_not_a_success(self):
        rows=[
            {'exact_match':True,'type_correct':True,'has_area':True,'candidate':'12.3','failure_cause':None},
            {'exact_match':False,'type_correct':True,'has_area':False,'candidate':None,'failure_cause':'no_area'},
            {'exact_match':None,'type_correct':True,'has_area':True,'candidate':'99.9','failure_cause':None},
        ]
        result=summarize(rows)
        self.assertEqual(result['scorable'],2)
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['incorrect'],1)
        self.assertEqual(result['unscored'],1)
        self.assertEqual(result['end_to_end_exact_match_rate'],.5)
        self.assertEqual(result['verified_correct_fraction_of_all_images'],1/3)
        self.assertEqual(result['failure_causes'],{'no_area':1})

    def test_no_confirmed_references_has_no_accuracy(self):
        result=summarize([{'exact_match':None,'type_correct':True,'has_area':True,'candidate':'12.3','failure_cause':None}])
        self.assertIsNone(result['end_to_end_exact_match_rate'])
        self.assertEqual(result['correct'],0)
        self.assertEqual(result['verified_correct_fraction_of_all_images'],0)


if __name__=='__main__':unittest.main()
