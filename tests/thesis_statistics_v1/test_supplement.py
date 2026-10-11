from pathlib import Path
import copy, json, tempfile, unittest
from scripts.thesis_statistics_v1.planning import riley_scenarios
from scripts.thesis_statistics_v1.export_aggregate import assert_aggregate
from scripts.thesis_statistics_v1.runtime import check_output_path

ROOT=Path(__file__).resolve().parents[2]
AGG=ROOT/'results/aggregate/10_statistical_supplement'

class SupplementTests(unittest.TestCase):
    def test_riley_replays_published_scenarios(self):
        expected=json.loads((AGG/'riley_scenarios.json').read_text(encoding='utf8'))
        a=expected['assumptions']
        actual=riley_scenarios(a['rate_per_person_year'],a['mean_observed_years'])
        for old,new in zip(expected['rows'],actual['rows']):
            self.assertEqual(old['minimum_for_scenario'],new['minimum_for_scenario'])
            self.assertEqual(old['n_optimism'],new['n_optimism'])
            self.assertEqual(old['n_risk_precision'],new['n_risk_precision'])
            for x,y in zip(old['risk_interval'],new['risk_interval']):self.assertAlmostEqual(x,y,places=14)

    def test_ipa_has_matching_null_denominator(self):
        j=json.loads((AGG/'baseline_and_ipa.json').read_text(encoding='utf8'))
        null=j['models']['null']['ibs']
        for value in j['models'].values():self.assertAlmostEqual(value['ipa'],1-value['ibs']/null,places=13)
        self.assertEqual(j['models']['null']['ipa'],0)

    def test_all_tau_contrasts_retained_and_adjusted_intervals_wider(self):
        j=json.loads((AGG/'tau_multiplicity.json').read_text(encoding='utf8'))
        self.assertEqual(set(j),{'cox','rsf','mri'})
        for value in j.values():
            d=value['difference'];self.assertEqual(d['comparisons_per_metric'],3)
            for metric in ['ibs','uno']:
                ordinary=d['delta_'+metric+'_ci95'];adjusted=d['delta_'+metric+'_bonferroni_ci']
                self.assertLessEqual(adjusted[0],ordinary[0]);self.assertGreaterEqual(adjusted[1],ordinary[1])

    def test_export_rejects_private_keys_and_paths(self):
        for value in [{'patient_hash':['abc']},{'x':'pt_0123456789abcdef'},{'x':'C:/private/a.csv'}]:
            with self.assertRaises(ValueError):assert_aggregate(value)
        assert_aggregate({'citation':'https://doi.org/10.1002/sim.7992','n':1059})
        for path in AGG.glob('*.json'):assert_aggregate(json.loads(path.read_text(encoding='utf8')))

    def test_output_cannot_enter_checkout_or_locked_results(self):
        with tempfile.TemporaryDirectory() as td:
            top=Path(td);repo=top/'release';project=top/'research'
            for bad in [repo/'private',project/'results/new',project,top]:
                with self.assertRaises(ValueError):check_output_path(bad,project,repo)
            self.assertEqual(check_output_path(top/'private_out',project,repo),(top/'private_out').resolve())

if __name__=='__main__':unittest.main()
