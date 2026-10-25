import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

script = Path(__file__).resolve().parents[1] / 'scripts/inspect_station_dataset.py'
spec = importlib.util.spec_from_file_location('inspect_station_dataset', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(image, format='JPEG')
        self.jpeg = image.getvalue()

    def write(self, episode, joints=8, cameras=2, task='place pepperoni on pizza'):
        rows = [dict(episode_start_ns=episode + 1,
                     timestamp_ns_since_episode_start=t,
                     joints=[dict(position_norm=0.5, goal_norm=0.5)] * joints,
                     images=[dict(jpeg=self.jpeg)] * cameras, task=task)
                for t in (0, 100_000_000, 200_000_000)]
        pq.write_table(pa.Table.from_pylist(rows), self.root / f'{episode}.parquet')

    def test_three_episode_directory_and_previews(self):
        for episode in range(3):
            self.write(episode)
        result = module.inspect_dataset(self.root, 3, 'place pepperoni on pizza')
        self.assertEqual(len(result), 3)
        self.assertTrue((self.root / 'inspection/0-camera-0.jpg').exists())

    def test_rejects_empty_or_incomplete_export(self):
        with self.assertRaises(ValueError):
            module.inspect_dataset(self.root, 3, 'place pepperoni on pizza')
        self.write(0)
        with self.assertRaises(ValueError):
            module.inspect_dataset(self.root, 3, 'place pepperoni on pizza')

    def test_rejects_wrong_joints_cameras_or_task(self):
        for overrides in (dict(joints=6), dict(cameras=1), dict(task='wrong task')):
            with self.subTest(overrides=overrides):
                self.write(0, **overrides)
                with self.assertRaises(ValueError):
                    module.inspect_dataset(self.root, 1, 'place pepperoni on pizza')

if __name__ == '__main__':
    unittest.main()
