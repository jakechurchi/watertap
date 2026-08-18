FILE DESCRIPTIONS:
- validation_plot.py => Function to compare the energy surrogates to the actual plant data

- Aug_21_kW_hourly.csv => total energy consumption from the modeled components of the treatment train (RO pumps, UF pumps, UV, decarb)
- Aug_21_real_operations.csv => Gives the flowrates 
- Aug_21_kW_hourly_WRD_model_validation.png => Uses data from two files above to create a validation plot using a month of data.
- Aug_21_kW_breakdown.csv => Gives the power consumption of each plant component instead of the total plant energy. (UNUSED)


NOTES:
- October 2021 data is also included
- week suffix reduces data down to one week
- The monthly data (Aug_21_kW_hourly.csv) is used for the validation plots contained in this folder only
- Aug_21_real_operations.csv format is meant to work with the JKM model to serve as an input fixing the operations of each hour