"""robustddg — robustness extensions to AutoDDG-style dataset description generation.

Two research prototypes live here:

1. Profiler trustworthiness: inject controlled errors into the statistical
   profile (`inject_errors`), measure how they propagate into the generated
   descriptions and retrieval, and gate the prompt with a cross-check between
   the LLM's column semantics and the statistics (`crosscheck`).

2. Adversarial robustness: poison uploaded tables with prompt injections or
   SEO keyword stuffing (`attacks`), measure rank inflation in a BM25 dataset
   search engine (`search`), and evaluate defenses (`defenses`).
"""
