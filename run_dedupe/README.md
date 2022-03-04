
# Running dedupe

Because our experiment involves running dedupe on data that already has ground truth labels, we were able to modify the dedupe function that handles accepting user input during active learning so that the correct label was automatically supplied. 

For example, we replaced the call to the `console_label()` function with a call to our modified `console_label_with_budget()` function. This allowed us to automate the process of a human suppling N labels. The main change involved is below. The only additional change was adding a stopping condition to stop after a certain number of labels had been supplied. The full function can be viewed in `modified_dedupe_functions.py`.

```python
    
    # normally if you are using dedupe, the person identifier (here called sid) would not
    # be known. but for our experimentation setup, it was. so we use it to determine
    # the correct user input to skip the required human input component of dedupe.
    if record_pair[0]['sid'] == record_pair[1]['sid']:
        user_input = 'y'
    else:
        user_input = 'n'

```

Other minor adjustments were made to the dedupe code for the purpose of returning more metrics and statistics about what was going on under the hood at various steps in the linking process. These edits did not impact how dedupe worked or performed when linking or deduplicating input datasets. We mention these edits simply as a warning that running our code exactly as is may result in errors if you don't make similar edits to dedupe code.
