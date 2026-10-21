import unittest
from softelo_release.data import battle_id
from softelo_release.export import sanitize, shared_eligible, paper_valid, arrow_schema


class AnnotationTests(unittest.TestCase):
    def row(self):
        return dict(model_a='a', model_b='b', human_pref=0., judge_pref=.75,
                    instruction_id='source-1', metadata={'original_question_id':'1'},
                    scores_a={'quality':8.}, scores_b={'quality':6.},
                    instruction='DO NOT EXPORT THIS TEXT', completion_a='PRIVATE RESPONSE')

    def test_judge_and_order_independent_battle_id(self):
        self.assertEqual(battle_id('x','1','a','b'), battle_id('x','1','b','a'))
        self.assertNotEqual(battle_id('x','1','a','b'), battle_id('y','1','a','b'))

    def test_export_allowlist_and_missing_orders(self):
        out = sanitize(self.row(),corpus='x',judge='judge',row_index=0,
                       source_index={'1':('a','b',0.)},revision='rev')
        self.assertEqual(set(out), set(arrow_schema().names))
        self.assertNotIn('instruction',out)
        self.assertNotIn('completion_a',out)
        self.assertFalse(out['has_order_scores'])
        self.assertIsNone(out['scores_a_original'])
        self.assertEqual(out['judge_pref'],.75)  # raw weak label retained

    def test_wrong_source_link_rejected(self):
        with self.assertRaises(ValueError):
            sanitize(self.row(),corpus='x',judge='j',row_index=0,
                     source_index={'1':('c','b',0.)},revision='rev')

    def test_paper_and_shared_filters_are_distinct(self):
        row=self.row();row['scores_a']['quality']=11.
        self.assertTrue(paper_valid(row))
        self.assertFalse(shared_eligible(row))
        row['scores_a']['quality']=None
        self.assertFalse(paper_valid(row))

    def test_source_orientation_checked(self):
        row=self.row();row.update(model_a='b',model_b='a',human_pref=1.)
        out=sanitize(row,corpus='x',judge='j',row_index=0,
                     source_index={'1':('a','b',0.)},revision='rev')
        self.assertTrue(out['source_order_reversed'])

class RevisionTests(unittest.TestCase):
    def test_mutable_remote_revision_rejected_before_network(self):
        from softelo_release.data import resolve_release
        with self.assertRaises(ValueError):
            resolve_release('example/nonexistent-release', revision='main')


if __name__ == '__main__':
    unittest.main()
