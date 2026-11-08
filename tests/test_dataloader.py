"""验证 NPZ 字段、类别映射、切片对应关系及测试病例过滤。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'torchvision'))
if HAS_DEPENDENCIES:
    import numpy as np
    import torch
    from Dataloader import dataset


@unittest.skipUnless(HAS_DEPENDENCIES, '需要 NumPy、PyTorch 和 torchvision')
class DataloaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cmr = np.arange(3 * 2 * 2 * 2, dtype=np.int16).reshape(3, 2, 2, 2)
        self.labels = np.tile(np.arange(4, dtype=np.uint8), 4).reshape(2, 2, 2, 2)

    def save_case(self, name, split='Testing', status='labeled', valid=True,
                  empty=False, cmr=None):
        cmr = self.cmr if cmr is None else cmr
        labels = self.labels[:0] if empty else self.labels
        info = {
            'schema': {'array_axes': {'cmr': ['T', 'Z', 'Y', 'X'],
                                      'labels': ['phase', 'Z', 'Y', 'X']}},
            'case': {'case_id': name, 'original_split': split,
                     'official_label_status': status},
            'annotation': {'time_indices': [] if empty else [0, 2],
                           'phase_names': [] if empty else ['ED', 'ES'],
                           'has_labels': not empty, 'phase_metadata_valid': valid},
            'source': {'files': {'cmr_4d': {'zooms': [1.5,1.2,10.,1.]}}},
            'processing': {'output': {'cmr_shape': list(cmr.shape),
                                      'labels_shape': list(labels.shape)}},
            'optional_metadata': None,
        }
        np.savez_compressed(self.root / f'{name}.npz', cmr=cmr, labels=labels,
                            info=json.dumps(info))

    def load(self, **kwargs):
        return dataset(data_path=self.root, **{'mode': 'all', 'datasets': 'mnms', **kwargs})

    def test_dimensions_normalization_and_default_collation(self):
        self.save_case('case')
        volume = self.load()[0]
        slices = self.load(dim='2D')
        self.assertEqual(set(volume), {'cmr', 'label', 'info'})
        self.assertEqual(volume['cmr'].shape, (2, 3, 128, 128))
        self.assertEqual(volume['label'].shape, (2, 2, 128, 128))
        self.assertEqual(volume['info']['source_spatial_shape'].tolist(), [2, 2])
        self.assertEqual(volume['info']['spatial_shape'].tolist(), [128, 128])
        self.assertEqual(volume['cmr'].dtype, torch.float32)
        self.assertEqual(volume['label'].dtype, torch.long)
        self.assertGreaterEqual(volume['cmr'].min().item(), 0)
        self.assertLessEqual(volume['cmr'].max().item(), 1)
        self.assertEqual(len(slices), 2)
        for s in range(2):
            torch.testing.assert_close(slices[s]['cmr'], volume['cmr'][s])
            torch.testing.assert_close(slices[s]['label'], volume['label'][s])
            self.assertEqual(slices[s]['info']['slice_index'], s)
        batch = next(iter(torch.utils.data.DataLoader(self.load(), batch_size=1)))
        self.assertEqual(batch['info']['filename'], ['case.npz'])
        self.assertIsNone(json.loads(batch['info']['metadata'][0])['optional_metadata'])
        torch.testing.assert_close(volume['info']['source_spacing_xyz'],torch.tensor([1.5,1.2,10.],dtype=torch.float64))
        torch.testing.assert_close(volume['info']['spacing_xyz'],torch.tensor([1.5*2/128,1.2*2/128,10.],dtype=torch.float64))
        torch.testing.assert_close(slices[0]['info']['spacing_xyz'],volume['info']['spacing_xyz'])

    def test_rectangular_crop_resize_preserves_physical_field(self):
        self.labels = np.zeros((2,2,80,100),dtype=np.uint8)
        self.save_case('rectangular',cmr=np.ones((3,2,80,100),dtype=np.float32))
        spacing = self.load()[0]['info']['spacing_xyz']
        torch.testing.assert_close(spacing,torch.tensor([1.171875,.75,10.],dtype=torch.float64))
        torch.testing.assert_close(spacing[:2]*128,torch.tensor([150.,96.],dtype=torch.float64))

    def test_class_mapping_matches_acdc(self):
        self.save_case('case')
        original = torch.from_numpy(self.labels).long().repeat_interleave(64, -2).repeat_interleave(64, -1)
        torch.testing.assert_close(self.load(datasets='acdc')[0]['label'],
                                   original.permute(1, 0, 2, 3))
        mapped = torch.tensor([0, 3, 2, 1])[original]
        torch.testing.assert_close(self.load()[0]['label'], mapped.permute(1, 0, 2, 3))
        self.assertEqual(self.load()[0]['info']['time_indices'].tolist(), [0, 2])

    def test_test_filter_handles_missing_unofficial_and_invalid_labels(self):
        self.save_case('a_valid')
        self.save_case('b_empty', empty=True)
        self.save_case('c_unofficial', status='unlabeled')
        self.save_case('d_invalid_phase', valid=False)
        self.save_case('e_training', split='Training/Labeled')
        selected = self.load(mode='test', dim='2D')
        self.assertEqual(selected.patients, ['a_valid.npz'])
        self.assertEqual(len(selected), 2)
        self.assertEqual(self.load(require_labels=True).patients,
                         ['a_valid.npz', 'e_training.npz'])

    def test_filter_does_not_change_existing_split(self):
        self.save_case('a_empty', empty=True)
        self.save_case('b_valid')
        self.save_case('c_valid')
        self.save_case('d_empty', empty=True)
        selected = self.load(mode='val', train_fraction=0.5, require_labels=True)
        self.assertEqual(selected.patients, ['c_valid.npz'])
        train = self.load(mode='train', train_fraction=0.5, dim='2D')
        self.assertEqual(train.patients, ['a_empty.npz', 'b_valid.npz'])
        self.assertEqual(len(train), 4)

    def test_empty_labels_and_constant_images(self):
        self.save_case('empty', empty=True, cmr=np.ones_like(self.cmr))
        volume = self.load()[0]
        self.assertEqual(volume['label'].shape, (2, 0, 128, 128))
        self.assertFalse(volume['info']['has_labels'])
        self.assertTrue(torch.equal(volume['cmr'], torch.zeros_like(volume['cmr'])))
        batch = next(iter(torch.utils.data.DataLoader(self.load(dim='2D'), batch_size=1)))
        self.assertEqual(batch['label'].shape, (1, 0, 128, 128))

    def test_downsampling_and_already_resized_inputs(self):
        for size in (256, 128):
            with self.subTest(size=size):
                self.labels = np.zeros((2, 2, size, size), dtype=np.uint8)
                self.labels[:, :, :size // 2, :size // 2] = 1
                self.labels[:, :, size // 2:, size // 2:] = 3
                image = np.broadcast_to(self.labels[:1], (3, 2, size, size)).copy()
                self.save_case('case', cmr=image)
                volume = self.load()[0]
                self.assertEqual(volume['cmr'].shape, (2, 3, 128, 128))
                self.assertEqual(volume['label'].shape, (2, 2, 128, 128))
                expected = torch.zeros((2, 2, 128, 128), dtype=torch.long)
                expected[:, :, :64, :64] = 3
                expected[:, :, 64:, 64:] = 1
                torch.testing.assert_close(volume['label'], expected)
                self.assertTrue(torch.all(volume['cmr'][..., 16, 16] > 0))
                self.assertTrue(torch.all(volume['cmr'][..., 112, 112] > 0.99))

    def test_invalid_annotation_index_is_rejected(self):
        self.save_case('bad', cmr=self.cmr[:2])
        with self.assertRaisesRegex(ValueError, '时间索引'):
            self.load()[0]

    def test_background_only_annotation_is_valid(self):
        self.labels.fill(0)
        self.save_case('background')
        selected = self.load(mode='test')
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0]['info']['valid_labels'])


if __name__ == '__main__':
    unittest.main()
