import pandas as pd
import numpy as np

def any_link_accuracy(exp, admin):
    '''Does the experiment record have a link when it should and not have a link when
    it should not?'''

    exp = exp.copy()
    admin = admin.copy()

    exp['link_expected'] = exp.sid.isin(admin.sid)
    exp['link_made'] = exp.cluster_id.isin(admin.cluster_id)
    
    result = exp.link_expected == exp.link_made
    result.name = 'accuracy'

    return result


def any_link_recall(exp, admin):
    '''For the experiment records that should have record(s) in admin, how many do?'''

    exp = exp.copy()
    admin = admin.copy()

    exp__link_expected = exp[exp.sid.isin(admin.sid)].copy()
    
    result = exp__link_expected.cluster_id.isin(admin.cluster_id)
    result.name = 'link_found_when_expected'

    return result


def any_link_precision(exp, admin):
    '''For the experiment records that link to record(s) in admin, how many should have?'''

    exp = exp.copy()
    admin = admin.copy()

    exp__link_made = exp[exp.cluster_id.isin(admin.cluster_id)].copy()
    
    result = exp__link_made.sid.isin(admin.sid)
    result.name = 'link_expected_when_found'

    return result


def any_link_tnr(exp, admin):
    '''For the experiment records that don't link to record(s) in admin, in how many 
    is that the correct decision?'''

    exp = exp.copy()
    admin = admin.copy()

    exp__no_link_expected = exp[exp.sid.isin(admin.sid) == False].copy()
    
    result = exp__no_link_expected.cluster_id.isin(admin.cluster_id) == False
    result.name = 'no_link_found_when_not_expected'

    return result


def correct_links_recall(exp, admin):
    '''For each experiment record that should have linked to record(s) in admin, did it link 
    to all the right things?

    NOTE: This doesn't need to be a left merge because the experiment records that
    *should* be in admin have SIDs in admin (and sid is the merge key).
    '''

    exp = exp.copy()
    admin = admin.copy()

    m = pd.merge(exp, admin[['sid', 'cluster_id']], on='sid', suffixes=['__exp', '__admin'])

    m['correct_link_found'] = (m.cluster_id__exp == m.cluster_id__admin).astype(int)
    result = m.groupby('sid').correct_link_found.all().astype(int)
    result.name = 'all_true_links_found'
    
    return result


def correct_links_precision(exp, admin):
    '''For each experiment record that linked to record(s) in admin, were all of the links correct?
    
    NOTE: This doesn't need to be a left merge because the each experiment record that *links to records
    in the admin dataset* have cluster_ids in admin.
    '''

    exp = exp.copy()
    admin = admin.copy()

    m = pd.merge(exp, admin[['sid', 'cluster_id']], on='cluster_id', suffixes=['__exp', '__admin'])

    m['predicted_link_correct'] = (m.sid__exp == m.sid__admin).astype(int)
    result = m.groupby('sid__exp').predicted_link_correct.all().astype(int)
    # QUESTION: group by sid__exp or cluster_id? I guess sid to keep at experiment record level (for NM this doesn't matter)
    result.name = 'all_predicted_links_correct'

    return result


def pair_level_recall(exp, admin):
    '''For each admin record that should have linked to an experiment record, did it link 
    to that experiment record?

    NOTE: This doesn't need to be a left merge because the admin records that
    *should* be in experiment dataset have SIDs in exp (and sid is the merge key).
    '''

    exp = exp.copy()
    admin = admin.copy()

    m = pd.merge(admin, exp[['sid', 'cluster_id']], on='sid', suffixes=['__admin', '__exp'])

    m['correct_link_found'] = (m.cluster_id__admin == m.cluster_id__exp).astype(int)
    
    return m['correct_link_found']


def pair_level_precision(exp, admin):
    '''For each admin record that linked to an experiment record, should it have?
    
    NOTE: This doesn't need to be a left merge because the each admin record that *links to records
    in the experiment dataset* have cluster_ids in admin.
    '''

    exp = exp.copy()
    admin = admin.copy()

    m = pd.merge(admin, exp[['sid', 'cluster_id']], on='cluster_id', suffixes=['__admin', '__exp'])

    m['predicted_link_correct'] = (m.sid__admin == m.sid__exp).astype(int)
    
    return m['predicted_link_correct']


def pair_level_tnr(exp, admin):
    '''For the admin records that don't link to record(s) in exp, in how many was that the correct 
    decision?'''

    exp = exp.copy()
    admin = admin.copy()

    admin__no_link_expected = admin[admin.sid.isin(exp.sid) == False].copy()
    
    result = admin__no_link_expected.cluster_id.isin(exp.cluster_id) == False
    result.name = 'no_link_found_when_not_expected'

    return result


def eval_predictions(df):

    exp = df[df.dataset == 'experiment_evaluation'].copy()
    admin = df[df.dataset == 'admin_evaluation'].copy()

    # just in case there are NAs
    exp.cluster_id.fillna(-1, inplace=True)
    admin.cluster_id.fillna(-2, inplace=True)

    tpr_any = any_link_recall(exp, admin).mean()
    prc_any = any_link_precision(exp, admin).mean()
    tnr_any = any_link_tnr(exp, admin).mean()
    acc_any = any_link_accuracy(exp, admin).mean()
    
    tpr_corr = correct_links_recall(exp, admin).mean()
    prc_corr = correct_links_precision(exp, admin).mean()

    tpr_pair = pair_level_recall(exp, admin).mean()
    prc_pair = pair_level_precision(exp, admin).mean()
    tnr_pair = pair_level_tnr(exp, admin).mean()

    fpr_any = 1 - tnr_any
    fnr_any = 1 - tpr_any
    error_any = fpr_any + fnr_any

    fpr_pair = 1 - tnr_pair
    fnr_pair = 1 - tpr_pair
    error_pair = fpr_pair + fnr_pair
    
    results = pd.DataFrame(
        [[tpr_any, prc_any, tnr_any, error_any, acc_any, tpr_corr, prc_corr, tpr_pair, prc_pair, tnr_pair, error_pair]], # single row
        columns=['tpr_any', 'prc_any', 'tnr_any', 'error_any', 'acc_any', 'tpr_corr', 'prc_corr', 'tpr_pair', 'prc_pair', 'tnr_pair', 'error_pair'])

    return(results)
