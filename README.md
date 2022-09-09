## Dedupe Paper

Replication code for dedupe paper.

#### Tasks in repo

1. **Clean OJIN data** - Drop columns from the raw data that we don't need, drop rows with bad person identifiers, create separate name columns from full-name, etc.
2. **Generate datasets** - Sample the clean OJIN data so it meets the specification(s) we want for the experiment (e.g. number of rows, share of matches that are exact, etc.)
3. **Run dedupe (for all specifications)** - Run our modified version of the dedupe package that automatically supplies the ground-truth label sought during active learning. 
4. **Run Name Match (for all specifications)** - Run Name Match, a supervised record linkage tool that leverages all available ground truth, as a point of comparison
5. **Run fastLink (for all specifications)** - Run fastLink, an unsupervised record linkage tool, as a point of comparison
6. **Get results file** - Combine the performance metrics from dedupe and Name Match, compute additional performance metrics on special subsets like non-exact pairs, etc.
7. **Create figures** - Generate figures for the paper
8. **Get significance results** - Output excel file with information about statistical significance

#### Running the code

We used a conda environment to ensure reproducibility of results across computing environments. To create and activate the environment we used, navigate to the dedupe_paper folder in a terminal and type the following commands:

```
# for python tasks
conda env create -f environment_py.yml
conda activate dedupe_paper_py
```

For R tasks, we used R version 3.5.2. To install fastlink, type the following commands:

```
R
install.packages("fastLink", repos="http://cran.r-project.org")
q()
```

#### A note about versions

The record linkage tools used in this paper -- dedupe, Name Match, and fastLink -- are open source packages that are actively being developed and improved. It is worth noting that the findings of this paper may not hold over time as the inner workings of these tools change. The paper's findings were based on the following versions of each record linkage tool:

* Dedupe: a [slightly modified](https://github.com/mmcneill/dedupe-fork/tree/dedupe-paper) v2.0.6
* Name Match: 1.1.0 
* fastLink: 0.6.0
