"""N1 controls: seed exclusion, scope, holdout boundary, selection and metrics."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tcn_moment import tcn_legacy_pilot as pilot
from tcn_moment.replicate_tcn import (
    DEFAULT, CONDITIONS, protocol, seed_config, validate_split, prediction_metrics,
    read_predictions, early_stop_replay, aggregate, length_edges, stratified_metrics,
)


class ReplicationTests(unittest.TestCase):
    def test_scope_rejects_pilot_test_expansion(self):
        p = protocol(DEFAULT)
        self.assertEqual(len(p['seeds']) * len(p['train_fractions']) * len(CONDITIONS), 36)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'protocol.json'
            for key in ('seed43_in_replication_mean', 'test_evaluated', 'automatic_next_stage',
                        'automatic_shutdown', 'pi_wake_enabled'):
                file.write_text(json.dumps({**p, key: True}))
                with self.assertRaises(ValueError): protocol(file)
            q = copy.deepcopy(p)
            q['seeds'] = [42, 43, 44, 45, 46]
            file.write_text(json.dumps(q))
            with self.assertRaises(ValueError): protocol(file)
            q = copy.deepcopy(p)
            q['diagnostics']['padding_widths'].append(2048)
            file.write_text(json.dumps(q))
            with self.assertRaises(ValueError): protocol(file)

    def test_seed_config_preserves_frozen_training(self):
        p = protocol(DEFAULT)
        parent = pilot.protocol(Path(p['parent_protocol']))
        changed = {'seed', 'output_dir', 'suite', 'split_manifest'}
        for seed in p['seeds']:
            cp = seed_config(p, seed)
            self.assertEqual(cp['seed'], seed)
            self.assertEqual({k:v for k,v in cp.items() if k not in changed},
                             {k:v for k,v in parent.items() if k not in changed})

    def test_split_validation_rejects_protocol_drift_and_overlap(self):
        parent = {'split_manifest': 'unused'}
        original = {'protocol': {'random_state':43,'dataset_sha256':'a'}}
        m = {'protocol': {'random_state':42,'dataset_sha256':'a'}, 'splits': {
            name: [f'{name}{i}' for i in range(n)] for name,n in
            [('train',24569),('validation',3510),('test',7020)]}}
        with patch.object(pilot, 'read', return_value=original):
            validate_split(m, parent, 42)
            bad = copy.deepcopy(m);bad['splits']['test'][0] = bad['splits']['train'][0]
            with self.assertRaisesRegex(ValueError, 'Overlapping'):validate_split(bad,parent,42)
            bad = copy.deepcopy(m);bad['protocol']['dataset_sha256'] = 'b'
            with self.assertRaisesRegex(ValueError, 'protocol'):validate_split(bad,parent,42)

    def test_metrics_battery_index_and_argmax(self):
        y = np.array([0,1,2,2,0,1])
        p = np.array([[.9,.05,.05],[.1,.8,.1],[.1,.1,.8],
                      [.7,.1,.2],[.1,.1,.8],[.1,.7,.2]])
        m = prediction_metrics(y,p)
        self.assertEqual(m['confusion_matrix'],[[1,0,1],[0,2,0],[1,0,1]])
        self.assertEqual(m['battery_recall'], .5)
        self.assertEqual(m['battery_FPR'], .25)
        self.assertAlmostEqual(m['macro_f1'], 2/3)

    def test_prediction_id_order_and_probabilities(self):
        v = {'ids':np.array(['001','002','003']), 'y':np.array([0,1,2])}
        p = np.eye(3)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'rows.csv'
            pilot.save_predictions(file,v['ids'],v['y'],p)
            got, _ = read_predictions(file,v,1.)
            np.testing.assert_array_equal(got,p)
            frame = pd.read_csv(file,dtype={'sample_id':str})
            frame.iloc[::-1].to_csv(file,index=False)
            with self.assertRaises(ValueError):read_predictions(file,v)
            pilot.save_predictions(file,v['ids'],v['y'],p)
            frame.loc[0,'p0'] = .8
            frame.to_csv(file,index=False)
            with self.assertRaises(ValueError):read_predictions(file,v)

    def test_early_stop_replay_is_prefix_only(self):
        scores = [.5]+[.50001]*7+[.9]*42
        history = [{'epoch':i+1,'macro_f1':x} for i,x in enumerate(scores)]
        result = early_stop_replay(history)
        self.assertEqual(result, {'stopped_epoch':8,'selected_epoch':1,'macro_f1':.5})
        self.assertEqual(max(scores),.9)

    def test_length_bins_do_not_depend_on_validation(self):
        train = np.array([18,30,60,100,200,300])
        edges = length_edges(train)
        v = np.array([1,10000])
        rows = stratified_metrics(np.array([0,2]),np.array([[1.,0,0],[0,0,1.]]),v,edges)
        self.assertEqual([r['n'] for r in rows],[1,0,1])
        self.assertIsNone(rows[1]['metrics'])
        # Missing classes remain in the fixed3-class Macro-F1 denominator.
        self.assertAlmostEqual(rows[0]['metrics']['macro_f1'],1/3)
        np.testing.assert_array_equal(length_edges(train),edges)

    def test_aggregate_rejects_pilot_and_missing_seeds(self):
        rows = [{'seed':s,'fraction':f,'condition':c,'selected':{'macro_f1':.8+.01*i}}
                for f in (.1,.4,1.) for s in (42,44,45,46)
                for i,c in enumerate(CONDITIONS)]
        tables,deltas = aggregate(rows)
        self.assertEqual(len(tables),9)
        self.assertEqual(len(deltas),9)
        self.assertTrue(all(d['positive_seeds']==4 for d in deltas))
        with self.assertRaises(ValueError):aggregate(rows[1:])
        with self.assertRaises(ValueError):aggregate(rows+[dict(rows[0],seed=43)])


if __name__ == '__main__':unittest.main()
