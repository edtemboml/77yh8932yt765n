create or replace model ai4f.aapl_model 
options
( model_type='linear_reg', 
  input_label_cols=['close'],
  data_split_method = 'seq', 
  data_split_eval_fraction = 0.3,
  data_split_col   = 'date'

)

as 

select 
date, 
close, 
open, 
high, 
low, 
oc_diff,
prev_day_h, 
prev_day_l,
prev_day_o,
prev_day_close, 
prev_day_oc,
trend_3_days
trend_3high_days, 
trend_3low_days, 
trend_3open_days,
trend_3oc_days

from 
ai4f.model_data