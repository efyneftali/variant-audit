# ACMG/AMP Variant Classification Criteria

> Condensed development sample summarizing the ACMG/AMP 2015 framework (Richards et al.).
> For the authoritative text, see the full guideline (acmg.net / PMC) and ClinGen gene-specific
> specifications. This file exists to exercise the retrieval pipeline, not as a clinical reference.

A variant is classified by collecting evidence criteria, each with a direction (pathogenic or
benign) and a strength (stand-alone, very strong, strong, moderate, or supporting), then combining
them into a 5-tier call.

## Pathogenic criteria

- **PVS1 (very strong):** Null variant (nonsense, frameshift, canonical +/-1 or 2 splice site,
  initiation codon, single or multi-exon deletion) in a gene where loss of function is a known
  mechanism of disease.
- **PS1 (strong):** Same amino acid change as a previously established pathogenic variant,
  regardless of the nucleotide change.
- **PS2 (strong):** De novo (maternity and paternity confirmed) in a patient with the disease and
  no family history.
- **PS3 (strong):** Well-established functional studies show a damaging effect on the gene or gene
  product.
- **PS4 (strong):** The variant is significantly more frequent in affected individuals than in
  controls.
- **PM1 (moderate):** Located in a mutational hot spot or a critical, well-established functional
  domain without benign variation.
- **PM2 (moderate):** Absent from population databases (gnomAD), or at extremely low frequency for
  a recessive disorder.
- **PM3 (moderate):** For recessive disorders, detected in trans with a pathogenic variant.
- **PM4 (moderate):** Protein length change from in-frame indels in a non-repeat region, or
  stop-loss variants.
- **PM5 (moderate):** Novel missense change at a residue where a different pathogenic missense
  change has been seen before.
- **PM6 (moderate):** Assumed de novo, without confirmation of paternity and maternity.
- **PP1 (supporting):** Cosegregation with disease in multiple affected family members.
- **PP2 (supporting):** Missense variant in a gene with a low rate of benign missense variation,
  where missense is a common disease mechanism.
- **PP3 (supporting):** Multiple lines of computational evidence support a deleterious effect
  (conservation, evolutionary, splicing).
- **PP4 (supporting):** Patient phenotype or family history is highly specific for a disease with a
  single genetic cause.

## Benign criteria

- **BA1 (stand-alone):** Allele frequency above 5 percent in population databases. A common variant
  is benign on this alone.
- **BS1 (strong):** Allele frequency greater than expected for the disorder.
- **BS2 (strong):** Observed in a healthy adult with full penetrance expected at an early age
  (homozygous for recessive, heterozygous for dominant, hemizygous for X-linked).
- **BS3 (strong):** Well-established functional studies show no damaging effect.
- **BS4 (strong):** Lack of segregation in affected members of a family.
- **BP1 (supporting):** Missense variant in a gene where only truncating variants cause disease.
- **BP3 (supporting):** In-frame indels in a repetitive region without known function.
- **BP4 (supporting):** Multiple lines of computational evidence suggest no impact.
- **BP7 (supporting):** Synonymous variant with no predicted splice impact and low conservation.

## Combining criteria into a classification

- **Pathogenic:** for example, 1 PVS1 plus at least one strong (PS), or two or more moderate (PM);
  or two or more strong; or one strong plus several moderate or supporting.
- **Likely Pathogenic:** a similar but weaker combination, one tier down from Pathogenic.
- **Benign:** BA1 alone, or two or more strong benign (BS).
- **Likely Benign:** one strong benign plus one supporting, or two or more supporting benign.
- **Uncertain Significance (VUS):** the evidence does not meet the thresholds above, or pathogenic
  and benign criteria contradict each other.
