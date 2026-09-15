NAMING CONVENTIONS:
- week or month suffix indicates length of data.
- Aug_21 or Oct_21 indicates the month and year of the data

FILE DESCRIPTIONS:

- validation_plot.py => Function to compare the energy surrogates to the actual plant data

- Aug_21_kW_month.csv => total energy consumption from the modeled components of the treatment train (RO pumps, UF pumps, UV, decarb)
- Aug_21_real_operations_month.csv => Gives the flowrates 
- Aug_21_kW_breakdown_week.csv => Gives the power consumption of each plant component instead of the total plant energy. (UNUSED)

- Aug_21_kW_month_WRD_model_validation.png => Uses data from two files above to create a validation plot using a month of data.

NOTES:
- October 2021 data is also included, but only data to create a weekly plot (UNUSED)
- The monthly data (Aug_21_kW_month.csv) is used for the validation plots contained in this folder only
- Aug_21_real_operations.csv format is meant to work with the JKM model to serve as an input fixing the operations of each hour
- This whole data structure should be rethought and probably redone to use the actual surrogates used in pricetaker!