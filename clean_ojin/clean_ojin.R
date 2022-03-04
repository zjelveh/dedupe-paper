
library(optparse)
library(data.table)
library(stringr)
library(lubridate)
library(stringdist)
options(scipen = 999)


arg_list <- list(
  make_option('--raw_data_file'),
  make_option('--output_file')
)
args <- parse_args(OptionParser(option_list=arg_list))


ojin = fread(args$raw_data_file)
ojin_reduced = ojin[, .(casnbr, casfildat, sid, name, dob, sex, race)]
ojin_reduced[, date_of_birth:=mdy(dob)]
ojin_reduced[, casfildat:=mdy(casfildat)]

ojin_reduced = ojin_reduced[name!='']
ojin_reduced[sid=='M', sid:=NA]
ojin_reduced = ojin_reduced[grepl(' ', name)]
ojin_reduced[race=='NULL', race:='']
ojin_reduced[, sid:=as.numeric(sid)]
ojin_reduced = ojin_reduced[!is.na(sid)]
ojin_reduced = ojin_reduced[sid>100]

ojin_collapsed = ojin_reduced[, .SD[1], by=c('sid', 'casnbr')]

ojin_collapsed = ojin_collapsed[order(sid, casnbr)]
ojin_collapsed[, rowid:=1:.N]

ojin_collapsed[, first_name:=str_split(name, ' ', simplify = T, n = 2)[2], by=rowid]
ojin_collapsed[, last_name:=str_split(name, ' ', simplify = T, n = 2)[1], by=rowid]
ojin_collapsed[, middle_name:=str_split(first_name, ' ', simplify = T, n = 2)[2], by=rowid]
ojin_collapsed[, first_name:=str_split(first_name, ' ', simplify = T, n = 2)[1], by=rowid]
ojin_collapsed[, dob:=NULL]

missing_strings = c('', 'NA', "NULL", 'NAN')
ojin_collapsed = ojin_collapsed[!first_name %in% missing_strings]
ojin_collapsed = ojin_collapsed[!last_name %in% missing_strings]

ojin_collapsed[, age := floor(time_length(difftime(ojin_collapsed$casfildat, ojin_collapsed$date_of_birth, units = 'weeks'), unit = 'year'))]

fwrite(ojin_collapsed, file=args$output_file)
