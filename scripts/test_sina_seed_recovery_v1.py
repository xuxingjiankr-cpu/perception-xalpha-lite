"""Real-entry recovery checks: changing a latest master must not change a run."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import collect_sina_research_daily_v1 as c
from test_sina_research_daily_v1 import CFG, SESSIONS, bar


class SeedRecoveryTests(unittest.TestCase):
    def test_real_entry_reuses_verified_files_under_new_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts/collect_ashare_research_daily.py").write_text("helper", encoding="utf-8")
            (root / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
            master = json.dumps({"securityId": "SH.600000", "pointInTimeMembership": True}) + "\n"
            (root / "master.jsonl").write_text(master, encoding="utf-8")
            (root / "archived_master.jsonl").write_text(master, encoding="utf-8")
            client = Mock()
            client.get.side_effect = ['var KLC_K2_sh600000="unused";',
                                     'var sh600000hfq={"total":1,"data":[{"d":"1900-01-01","f":"2"}]};']
            def git(command, **kwargs):
                return "test_commit\n" if command[1] == "rev-parse" else ""
            args = argparse.Namespace(config="config.json", run_id="origin", resume=False, symbols=None, seed_run=None)
            with patch.object(c, "ROOT", root), patch.object(c.subprocess, "check_output", side_effect=git), \
                 patch.object(c, "PublicClient", return_value=client), \
                 patch.object(c, "decode_bars", return_value=[bar(dt) for dt in SESSIONS]), patch("builtins.print"):
                self.assertEqual(c.run(args), 0)
                origin = root / CFG["outputRoot"] / "origin"
                original_manifest = (origin / "manifest.json").read_bytes()
                original_record = (origin / "symbols/SH_600000.json").read_bytes()
                (root / "master.jsonl").write_text(master.replace("600000", "600004"), encoding="utf-8")
                args.resume = True
                with self.assertRaisesRegex(ValueError, "resume_contract_mismatch"):
                    c.run(args)
                recovery = {**CFG, "masterPaths": ["archived_master.jsonl"]}
                (root / "recovery.json").write_text(json.dumps(recovery), encoding="utf-8")
                args = argparse.Namespace(config="recovery.json", run_id="recovery", resume=False, symbols=None, seed_run="origin")
                self.assertEqual(c.run(args), 0)
                result = json.loads((root / CFG["outputRoot"] / "recovery/result.json").read_text())
                self.assertEqual(result["reusedSymbols"], 1)
                self.assertEqual(client.get.call_count, 2)
                self.assertEqual((origin / "manifest.json").read_bytes(), original_manifest)
                self.assertEqual((origin / "symbols/SH_600000.json").read_bytes(), original_record)
                source = root / CFG["dataRoot"] / "origin/raw/SH_600000.jsonl"
                copied = root / CFG["dataRoot"] / "recovery/raw/SH_600000.jsonl"
                self.assertEqual(source.read_bytes(), copied.read_bytes())
                copied.write_text("tampered", encoding="utf-8")
                self.assertNotEqual(source.read_bytes(), copied.read_bytes())
                args.resume = True
                self.assertEqual(c.run(args), 2)
                result = json.loads((root / CFG["outputRoot"] / "recovery/result.json").read_text())
                self.assertIn("resume_artifact_hash_mismatch", result["failClosedReason"])
                self.assertEqual(client.get.call_count, 2)

    def test_seed_rejects_changed_definition_master_or_access_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = root / CFG["outputRoot"] / "origin"
            origin.mkdir(parents=True)
            contract = {"symbols": ["SH.600000"], "sessionsSha256": "s", "decoderSha256": "d",
                        "atomicHelperSha256": "h", "dependencies": {}, "normalizerSha256": "n",
                        "masterSources": [{"sha256": "m"}], "codeCommit": "c"}
            manifest = {"config": CFG, "contract": contract, "researchOnly": True, "orders": [], "mayPromote": False}
            (origin / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (origin / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            with patch.object(c, "ROOT", root):
                self.assertTrue(c.seed_contract("origin", CFG, contract)["masterContentIdentical"])
                with self.assertRaisesRegex(ValueError, "seed_config_mismatch"):
                    c.seed_contract("origin", {**CFG, "rawVwapRelativeTolerance": .2}, contract)
                with self.assertRaisesRegex(ValueError, "seed_master_content_changed"):
                    c.seed_contract("origin", CFG, {**contract, "masterSources": [{"sha256": "changed"}]})
                with self.assertRaisesRegex(ValueError, "seed_normalizer_changed"):
                    c.seed_contract("origin", CFG, {**contract, "normalizerSha256": "changed"})
                (origin / "status.json").write_text(json.dumps({"failClosedReason": "AccessDenied:provider_access_denied_http_403"}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "access_denied_seed_cannot_restart"):
                    c.seed_contract("origin", CFG, contract)


if __name__ == "__main__":
    unittest.main()
