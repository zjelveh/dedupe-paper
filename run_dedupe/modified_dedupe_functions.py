

def console_label_with_budget(deduper: dedupe.api.ActiveMatching, budget=0, verbose=False) -> None:  # pragma: no cover
    '''
   Train a matcher instance (Dedupe, RecordLink, or Gazetteer) from the command line.
   Example

   .. code:: python

      > deduper = dedupe.Dedupe(variables)
      > deduper.prepare_training(data)
      > dedupe.console_label(deduper)
    '''

    finished = False
    use_previous = False
    fields = unique(field.field
                    for field
                    in deduper.data_model.primary_fields)

    buffer_len = 1  # Max number of previous operations
    examples_buffer: List[Tuple[TrainingExample, Literal['match', 'distinct', 'uncertain']]] = []
    uncertain_pairs: List[TrainingExample] = []

    label_info_rows = []

    ix = 0
    while budget > 0 :
        ix = ix + 1
        just_queried_for_uncertain_pairs = False
        try:
            if not uncertain_pairs:
                uncertain_pairs, uncertainties, stats = deduper.uncertain_pairs()
                just_queried_for_uncertain_pairs = True

            record_pair = uncertain_pairs.pop()
            if verbose:
                logger.info(f"{record_pair[0]['first_name']} {record_pair[0]['last_name']} {record_pair[0]['date_of_birth']} <==> {record_pair[1]['first_name']} {record_pair[1]['last_name']} {record_pair[1]['date_of_birth']}")
            uncertainties = uncertainties.pop()
        except IndexError:
            logger.info("COULD NOT GET ANY MORE UNCERTAIN PAIRS.")
            break

        # Automatically return the correct label
        if record_pair[0]['sid'] == record_pair[1]['sid']:
            user_input = 'y'
        else:
            user_input = 'n'
        budget = budget - 1

        em = 0
        if ((record_pair[0]['first_name'] == record_pair[1]['first_name']) and
           (record_pair[0]['last_name'] == record_pair[1]['last_name']) and
           (record_pair[0]['date_of_birth'] == record_pair[1]['date_of_birth'])):
           em = 1

        this_res = {
            'label_ix':ix,
            'label':user_input,
            'rowid_1':record_pair[0]['rowid'],
            'rowid_2':record_pair[1]['rowid'],
            'exact_match':em,
            'just_queried_for_uncertain_pairs':just_queried_for_uncertain_pairs,
            'match_phat':uncertainties[0],
            'block_yhat':uncertainties[1],
        }
        this_res.update(stats)
        label_info_rows.append(this_res.copy())

        ###

        if user_input == 'y':
            examples_buffer.insert(0, (record_pair, 'match'))
        elif user_input == 'n':
            examples_buffer.insert(0, (record_pair, 'distinct'))

        if budget == 0:
            print('Finished labeling', file=sys.stderr)

        if len(examples_buffer) > buffer_len:
            record_pair, label = examples_buffer.pop()
            if label in {'distinct', 'match'}:

                examples: TrainingData
                examples = {'distinct': [],
                            'match': []}
                examples[label].append(record_pair)  # type: ignore
                deduper.mark_pairs(examples)

    for record_pair, label in examples_buffer:
        if label in ['distinct', 'match']:

            exmples: TrainingData
            examples = {'distinct': [], 'match': []}
            examples[label].append(record_pair)  # type: ignore
            deduper.mark_pairs(examples)

    label_info = pd.DataFrame.from_records(label_info_rows)

    return label_info
