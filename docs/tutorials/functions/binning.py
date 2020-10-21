import numpy as np
import math
from sklearn.preprocessing import KBinsDiscretizer

def KBinsDiscretize(data_x, n_bins=0, encode="ordinal", strategy="uniform"):
   # Makes n_bins optional, calculates optimal n_bins by default
   # Sturge's Rule - num_bins = 1+log_2(num_inputs)
   if n_bins == 0:
      n_bins = math.floor(1 + math.log(data_x.shape[0], 2))
   binner = KBinsDiscretizer(n_bins, encode="ordinal", strategy="uniform")
   binner.fit(data_x)
   binned_x = binner.transform(data_x)
   return binned_x
