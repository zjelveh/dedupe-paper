## Dedupe Paper

Replication code for dedupe paper.


Tasks in repo

1. **Clean OJIN data** - Drop columns from the raw data that we don't need, drop rows with bad person identifiers, create separate name columns from full-name, etc.
2. **Generate datasets** - Sample the clean OJIN data so it meets the specification(s) we want for the experiment (e.g. number of rows, share of matches that are exact, etc.)
3. **Run dedupe (for all specifications)** - Run our modified version of the dedupe package that automatically supplies the ground-truth label sought during active learning. 
4. **Run Name Match (for all specifications)** - Run Name Match, a supervised record linkage tool that leverages all available ground truth, as a comparison
5. **Get results file** - Combine the performance metrics from dedupe and Name Match, compute additional performance metrics on special subsets like non-exact pairs, etc.
6. **Create figures** - Generate figures for the paper
