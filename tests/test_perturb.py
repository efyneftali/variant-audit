"""Tests for the VA-13 paraphrase/formatting-noise perturbation layer
(evals/perturb.py). Facts must survive untouched -- only phrasing changes --
and perturbation must be deterministic, or a robustness run isn't reproducible.
"""

import random

from evals import perturb


class TestPerturbClinvar:
    def test_preserves_found_and_variant_exactly(self):
        record = {
            "found": True,
            "variant": "rs28897696",
            "matches": [
                {"clinical_significance": "Pathogenic", "hgvs": "BRCA1:c.123A>G", "review_status": "reviewed by expert panel"},
            ],
        }
        out = perturb._perturb_clinvar(record, random.Random("seed"))
        assert out["found"] is True
        assert out["variant"] == "rs28897696"

    def test_paraphrases_stay_within_the_meaning_preserving_options(self):
        record = {
            "found": True,
            "matches": [{"clinical_significance": "Pathogenic", "hgvs": "x", "review_status": "reviewed by expert panel"}],
        }
        rng = random.Random("fixed")
        out = perturb._perturb_clinvar(record, rng)
        sig = out["matches"][0]["clinical_significance"].strip().lower()
        # every option in the table maps back to the same 3-tier bucket -- must
        # contain "pathogenic" and not silently become "benign"/"uncertain"
        assert "pathogenic" in sig
        assert "benign" not in sig

    def test_unknown_value_passes_through_unparaphrased(self):
        # a value outside the closed vocabulary must not crash or vanish --
        # formatting noise (case/whitespace) may still apply on top of it
        record = {"found": True, "matches": [{"clinical_significance": "some future ClinVar tier", "hgvs": "x", "review_status": "y"}]}
        out = perturb._perturb_clinvar(record, random.Random("seed"))
        # strip ALL whitespace before comparing -- formatting noise may insert
        # a stray space mid-word, so word-splitting isn't a safe comparison
        no_space = "".join(out["matches"][0]["clinical_significance"].lower().split())
        assert no_space == "somefutureclinvartier"

    def test_does_not_mutate_the_input_record(self):
        record = {"found": True, "matches": [{"clinical_significance": "Benign", "hgvs": "x", "review_status": "y"}]}
        perturb._perturb_clinvar(record, random.Random("seed"))
        assert record["matches"][0]["clinical_significance"] == "Benign"  # original untouched


class TestPerturbEnsembl:
    def test_preserves_numeric_and_structural_fields(self):
        record = {
            "found": True, "variant": "rs1", "chrom": "14", "start": 100, "end": 100,
            "ref": "T", "alt": "C", "allele_string": "T/C",
            "gene_symbol": "SERPINA1", "impact": "MODERATE", "most_severe_consequence": "missense_variant",
        }
        out = perturb._perturb_ensembl(record, random.Random("seed"))
        assert out["chrom"] == "14" and out["start"] == 100 and out["end"] == 100
        assert out["ref"] == "T" and out["alt"] == "C"

    def test_not_found_record_is_untouched(self):
        record = {"found": False}
        out = perturb._perturb_ensembl(record, random.Random("seed"))
        assert out == {"found": False}


class TestPerturbAlphamissense:
    def test_not_applicable_record_is_untouched(self):
        record = {"found": False, "applicable": False, "variant": "rs1"}
        out = perturb._perturb_alphamissense(record, random.Random("seed"))
        assert out == record

    def test_am_class_stays_within_meaning(self):
        record = {"found": True, "applicable": True, "am_class": "likely_pathogenic", "am_pathogenicity": 0.9}
        out = perturb._perturb_alphamissense(record, random.Random("seed"))
        assert "pathogenic" in out["am_class"].lower()
        assert out["am_pathogenicity"] == 0.9  # numeric score untouched


class TestDeterminism:
    def test_same_tool_and_variant_always_perturbs_the_same_way(self):
        reader_a = perturb._make_perturbed_reader("clinvar")
        reader_b = perturb._make_perturbed_reader("clinvar")

        fixture = {"found": True, "matches": [{"clinical_significance": "Pathogenic", "hgvs": "x", "review_status": "reviewed by expert panel"}]}
        import evals.replay as replay
        orig = replay.load_fixture
        replay.load_fixture = lambda tool, variant: fixture
        try:
            first = reader_a("rs123")
            second = reader_b("rs123")
        finally:
            replay.load_fixture = orig

        assert first == second  # deterministic per (tool, variant), not per call


class TestInstall:
    def test_rebinds_all_five_tool_names_on_the_graph_module(self):
        from src.variant_audit import graph
        from evals import replay

        saved = {attr: getattr(graph, attr) for _, (attr, _real) in replay._TOOLS.items()}
        try:
            perturb.install()
            for tool, (attr, _real) in replay._TOOLS.items():
                assert getattr(graph, attr).__name__ == f"perturbed_{tool}"
        finally:
            for attr, fn in saved.items():
                setattr(graph, attr, fn)
