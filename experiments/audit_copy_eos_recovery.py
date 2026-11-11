"""Restore a missing import binding for the preserved EOS-study audit."""
from pathlib import Path

import torch

import audit_copy_eos_training as auditor
import encoder_decoder_contract
from copy_eos_training import ROOT,DIRECTORY
from position_distractors import digest
from recurrent_retrieval import write


if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    target=DIRECTORY/'evaluation_recovery'
    protocol=ROOT/'docs/protocols/copy-eos-audit-binding.md'
    assert protocol.exists() and not (target/'audit.json').exists() and not hasattr(auditor,'Model')
    auditor.Model=encoder_decoder_contract.Model
    assert auditor.Model is encoder_decoder_contract.Model
    auditor.DIRECTORY=target
    paths=(Path(__file__),protocol,ROOT/'experiments/audit_copy_eos_training.py',target/'report.json')
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    auditor.main()
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    write(target/'audit_binding.json',dict(passed=True,hashes=hashes,
        model_binding='encoder_decoder_contract.Model',only_missing_import_and_artifact_directory_bound=True,
        training_updates_added=0,audit_sha256=digest(target/'audit.json')))
