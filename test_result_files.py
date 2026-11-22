import gzip
import json
import tempfile
import unittest
from pathlib import Path

from check_study import verify


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)/"study.json"
        self.case = dict(rate=12, trial=1, replicas=1, pages=16, router="rr")
        self.rows = [dict(id=i, cohort="cold", generated=2, arrival_ms=i*10,
                          wait_ms=1, ttft_ms=2, max_gap_ms=3, slo="met") for i in range(300)]
        self.summary = dict(**self.case, completed=300, per_replica=[300], slo_met=300,
            slo_breakdown=dict(met=300, ttft_only=0, gap_only=0, both=0), elapsed_ms=3000,
            goodput_req_s=100, ttft_p95_ms=2, cold_wait_p95_ms=1, trace_sha256="same")
        self.path.write_text(json.dumps(dict(cases=[self.summary])))

    def save_rows(self, rows):
        with gzip.open(self.path.with_suffix(".requests.jsonl.gz"), "wt", encoding="utf8") as out:
            out.write(json.dumps(dict(case=self.case,requests=rows))+"\n")

    def test_recomputes_all_labels(self):
        self.save_rows(self.rows)
        self.assertEqual(verify(self.path),300)
        self.rows[8]["max_gap_ms"] = 101
        self.save_rows(self.rows)
        with self.assertRaisesRegex(ValueError,"SLO label"):
            verify(self.path)

    def test_detects_missing_and_reordered_requests(self):
        self.save_rows(self.rows[:-1])
        with self.assertRaisesRegex(ValueError,"incomplete"):
            verify(self.path)
        self.rows.reverse()
        self.save_rows(self.rows)
        with self.assertRaisesRegex(ValueError,"identity"):
            verify(self.path)
