library(data.table)
library(fastLink)
library(optparse)
library(yaml)
options(scipen = 999)

arg_list <- list( 
  make_option('--dataset_dir'),
  make_option('--config_file'),
  make_option('--ss'),
  make_option('--em'),
  make_option('--type'),
  make_option('--iter'),
  make_option('--output_dir')
)
args <- parse_args(OptionParser(option_list=arg_list))

params = yaml.load_file(args$config_file)

ss = args$ss
em = args$em
type = args$type
iter = args$iter

blocks_file = paste0(args$output_dir, '/blocks__sst_', ss, '__emt_', em, '__type_', type, '___iter_', iter, '.csv')
predicted_matches_file = paste0(args$output_dir, '/predicted_matches__sst_', ss, '__emt_', em, '__type_', type, '__iter_', iter, '.csv')

runtime_file = paste0(args$output_dir, '/runtime.csv')

clean_data <- function(datum) {
    datum[, age_in_2020 := as.numeric(age_in_2020)]
    datum[, yob := as.numeric(substring(date_of_birth, 1, 4))]
    datum[, mob := substring(date_of_birth, 6, 7)]
    datum[, dob := substring(date_of_birth, 9, 10)]
    return(datum)
}

block <- function(dfA, dfB, min_block_size_before_join, blocks_file) {

    # block on last name (2 clusters via kmeans) and year of birth (+/- 1 year)
    raw_blocks = blockData(
        dfA = dfA, dfB = dfB, 
        varnames = c('last_name', 'yob'), 
        kmeans.block = c('last_name'), nclusters = 2,
        window.block = "yob", window.size = 1,
        n.cores = 8
    )

    # determine which blocks are too small and should be combined
    small_blocks_to_join= c()
    big_blocks_fine_as_is = c()
    for (i in 1:length(raw_blocks)) {
        if(length(raw_blocks[[i]]$dfA.inds) <= min_block_size_before_join){
            small_blocks_to_join = c(small_blocks_to_join, i)
        } else{
            big_blocks_fine_as_is = c(big_blocks_fine_as_is, i)
        }
    }

    # convert raw_blocks to blocks by combining small blocks
    blocks = list()
    for (k in big_blocks_fine_as_is) {
        block_i = length(blocks) + 1
        blocks[[block_i]] = list()
        blocks[[block_i]]$dfA.inds = raw_blocks[[k]]$dfA.inds
        blocks[[block_i]]$dfB.inds = raw_blocks[[k]]$dfB.inds
    }
    dfAs = c()
    dfBs = c()
    for (j in small_blocks_to_join) {
        dfAs = c(dfAs, raw_blocks[[j]]$dfA.inds)
        dfBs = c(dfBs, raw_blocks[[j]]$dfB.inds)
    }
    block_i = length(blocks) + 1
    blocks[[block_i]] = list()
    blocks[[block_i]]$dfA.inds = unique(dfAs)
    blocks[[block_i]]$dfB.inds = unique(dfBs)
    names(blocks) = paste0('block.', 1:length(blocks))

    # create a df of records and their blocks (for debugging / sanity checking)
    block_list = list()
    for (i in 1:length(blocks)) { 
        block_list[[i]] = rbind(
            data.table(
                block = names(blocks)[i],
                rowid = dfA[blocks[[paste0('block.', i)]]$dfA.inds]$rowid,
                yob = dfA[blocks[[paste0('block.', i)]]$dfA.inds]$yob,
                dataset = dfA[blocks[[paste0('block.', i)]]$dfA.inds]$dataset), 
            data.table(
                block = names(blocks)[i],
                rowid = dfB[blocks[[paste0('block.', i)]]$dfB.inds]$rowid,
                yob = dfB[blocks[[paste0('block.', i)]]$dfB.inds]$yob,
                dataset = dfB[blocks[[paste0('block.', i)]]$dfB.inds]$dataset))
    }
    blocks_df = rbindlist(block_list)
    blocks_df = blocks_df[, .(block, dataset, rowid, yob)]
    fwrite(blocks_df, blocks_file)

    return(blocks)

}

run_fastlink <- function(dfA, dfB) {

    out = tryCatch(
        {
            fastLink(
                dfA, dfB,
                varnames = c('first_name', 'last_name', 'middle_initial', 'date_of_birth', 'race', 'sex', 'mob', 'dob'),
                stringdist.match = c('first_name', 'last_name', 'date_of_birth'),
                partial.match = c('first_name', 'last_name', 'date_of_birth'), 
                dedupe.matches = F, n.cores = 8, threshold.match = .85, tol.em = 0.00001)
        },
        error=function(e) {
            message("Error running fastLink function:")
            message(e)
            message('\n')
            return(NA)
        },
        finally={
        }
    )
    return(out)
}

get_predicted_matches <- function(dfA, dfB, fl_results) {

    predicted_matches = data.table(
        rowid1 = dfA[fl_results$matches$inds.a]$rowid,
        sid1 = dfA[fl_results$matches$inds.a]$sid,
        d1 = dfA[fl_results$matches$inds.a]$dataset,
        rowid2 = dfB[fl_results$matches$inds.b]$rowid,
        sid2 = dfB[fl_results$matches$inds.b]$sid,
        d2 = dfB[fl_results$matches$inds.b]$dataset
    )

    return(predicted_matches)

}

admin_file = paste0(args$dataset_dir, '/admin_evaluation__', ss, '__', em, '.csv')
experi_file = paste0(args$dataset_dir, '/experiment_evaluation__', ss, '__', em, '.csv')

if (file.exists(admin_file) & file.exists(experi_file)) {

    admin = fread(admin_file)
    experi = fread(experi_file)

    admin = clean_data(admin)
    experi = clean_data(experi)

    print(paste0(type, " (n admin rows: ", ss, "; share exact match: ", em, ")"))

    if (type == 'record_linkage') { 

        start_block = Sys.time()
        blocks = block(experi, admin, params$min_block_size_before_join[[type]], blocks_file)
        end_block = Sys.time()
        
        start_fl = Sys.time()
        all_fl_results = list() # store up results from running fastLink on all blocks
        all_predicted_matches = list()
        for (i in 1:length(blocks)) {

            dfA = experi[blocks[[i]]$dfA.inds]
            dfB = admin[blocks[[i]]$dfB.inds]

            print(c(i, length(blocks), nrow(dfA), nrow(dfB)))

            if ((nrow(dfA) > 0) & (nrow(dfB) > 0)) {
                fl_results = run_fastlink(dfA, dfB)
                if (!is.na(fl_results)) {
                    all_fl_results[[length(all_fl_results)+1]] = fl_results
                } else {
                    next
                }
            }
            
            if (length(fl_results$matches$inds.a) > 0) {
                predicted_matches = get_predicted_matches(dfA, dfB, fl_results)
                all_predicted_matches[[length(all_predicted_matches)+1]] = predicted_matches
            }
            gc()
            
            if (nrow(dfA) < 5) {
              print("N predicted matches")
              print(nrow(predicted_matches))
            }
            
        }
        end_fl = Sys.time()

        predicted_matches = rbindlist(all_predicted_matches)
        print(nrow(predicted_matches))
        predicted_matches = unique(predicted_matches)
        print(nrow(predicted_matches))
        
        fwrite(predicted_matches, predicted_matches_file)
        
    }

    if (type == 'deduplication') {

        all_data = rbind(admin, experi)

        start_block = Sys.time()
        blocks = block(all_data, all_data, params$min_block_size_before_join[[type]], blocks_file)
        end_block = Sys.time()

        start_fl = Sys.time()
        all_fl_results = list() # store up results from running fastLink on all blocks
        all_predicted_matches = list()
        for (i in 1:length(blocks)) {

            dfA = all_data[blocks[[i]]$dfA.inds]
            dfB = all_data[blocks[[i]]$dfB.inds]

            print(c(i, length(blocks), nrow(dfA), nrow(dfB)))

            if (nrow(dfA) > 0) {
                fl_results = run_fastlink(dfA, dfB)
                if (!is.na(fl_results)) {
                  all_fl_results[[length(all_fl_results)+1]] = fl_results
                } else {
                  next
                }
            }

            if (length(fl_results$matches$inds.a) > 0) {
                predicted_matches = get_predicted_matches(dfA, dfB, fl_results)
                all_predicted_matches[[length(all_predicted_matches)+1]] = predicted_matches
            }
            gc()
        }
        end_fl = Sys.time()

        predicted_matches = rbindlist(all_predicted_matches)
        predicted_matches = predicted_matches[rowid1 < rowid2] # get rid of self-pairs (A-A) and reverse pairs (A-B and B-A)
        # predicted_matches = predicted_matches[d1 != d2] # DON'T DO THIS IF WE WANT CLUSTERING TO ALLOW FOR 2nd-DEGREE LINKS    
        print(nrow(predicted_matches))
        predicted_matches = unique(predicted_matches)
        print(nrow(predicted_matches))
        fwrite(predicted_matches, predicted_matches_file)
        
    }

    runtime_stats = data.table(
        # params
        ss = ss,
        em = em,
        framework = type,
        model_iter = iter,
        # runtime_stats
        runtime_min__block = difftime(end_block, start_block, units = "mins"),
        runtime_min__fl = difftime(end_fl, start_fl, units = "mins"),
        # timestamp
        timestamp = Sys.time()
    )
    fwrite(runtime_stats, runtime_file, append = TRUE)

}
