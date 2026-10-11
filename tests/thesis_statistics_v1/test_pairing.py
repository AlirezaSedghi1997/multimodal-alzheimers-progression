"""Synthetic checks of the reused paired-statistics routine; no study data."""
import copy, unittest
from scripts.xyz_fusion_v2.reporting import paired_ci

class PairingTests(unittest.TestCase):
    def cells(self):
        train={'event':[i%3!=0 for i in range(1,61)],'time':[float(i) for i in range(1,61)]}
        arm={'subjects':['synthetic_'+str(i) for i in range(12)],
             'event':[i%2==0 for i in range(12)],
             'time':[float(5+3*i) for i in range(12)],
             'curves':[[.95-.01*i,.85-.015*i,.75-.02*i] for i in range(12)],
             'censor_train':train}
        return [{'fold':0,'arms':{'base':arm,'same':copy.deepcopy(arm)}}]

    def test_identical_predictions_have_exact_zero_paired_difference(self):
        r=paired_ci(self.cells(),'base','same',[6.,12.,18.],20261011,40,3)
        self.assertEqual(r['delta_ibs_mean'],0.)
        self.assertEqual(r['delta_uno_mean'],0.)
        self.assertEqual(r['delta_ibs_bonferroni_ci'],[0.,0.])
        self.assertEqual(r['delta_uno_bonferroni_ci'],[0.,0.])

    def test_permuted_membership_is_rejected_before_resampling(self):
        cells=self.cells();cells[0]['arms']['same']['subjects'].reverse()
        with self.assertRaisesRegex(ValueError,'UNMATCHED_PAIRED_CONTRAST'):
            paired_ci(cells,'base','same',[6.,12.,18.],20261011,40,3)

if __name__=='__main__':unittest.main()
