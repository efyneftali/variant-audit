"""AlphaMissense tool — DeepMind pathogenicity prediction (missense variants only).

Source: precomputed predictions (CC-BY) on Zenodo / DeepMind download site
(https://github.com/google-deepmind/alphamissense). 71M variants — download the
table once into data/alphamissense/ and serve as a fast local lookup.

IMPORTANT: this is a SOTA *computational evidence input*, never ground truth
(grading against it would be grading the model against another model).
Only covers missense — return a clear "not applicable" for other variant types.

TODO(day-7): implement as a local table lookup.
"""


def get_alphamissense_score(variant: str) -> dict:
    """Return the AlphaMissense pathogenicity score + class for a missense variant."""
    raise NotImplementedError("TODO(day-7): look up the variant in the local AlphaMissense table")
