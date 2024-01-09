select * 
from 
ml.predict(MODEL ai4f.aapl_model, 
(select * from model_data where date >= '2019-01-01')
)