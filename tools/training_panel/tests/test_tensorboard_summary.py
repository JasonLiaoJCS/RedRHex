import tempfile
import time
import unittest
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tools.training_panel.training_panel.tensorboard_summary import ensure_tensorboard_summary


class TensorboardSummaryTests(unittest.TestCase):
    def test_ensure_tensorboard_summary_writes_png_from_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            writer = EventFileWriter(str(log_dir))
            for step, value in enumerate([1.0, 2.0, 3.5], start=1):
                writer.add_event(
                    Event(
                        wall_time=time.time(),
                        step=step,
                        summary=Summary(value=[Summary.Value(tag="Train/mean_reward", simple_value=value)]),
                    )
                )
            writer.close()

            summary = ensure_tensorboard_summary(log_dir, title="unit run")

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary.suffix, ".png")
            self.assertTrue(summary.is_file())
            self.assertGreater(summary.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
