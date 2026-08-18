FILE DESCRIPTIONS:
- energy_cost_calc.py => Simple function that takes an hourly profile and calculates the electricity bill using the 2023 TOU3 rate structure. This used to find the electricity costs of the "real operations", which is just a the hourly energy consumption profile.
- plot_real_ops.py => Script used to create the 


- Aug_kW_hourly_week.csv => total energy consumption from the modeled components of the treatment train (RO pumps, UF pumps, UV, decarb)
- Aug_real_operation.csv => flowrate percentages for each RO train with a 
- Aug_real_week.png => plot of the 
- wrd_result_Aug_optimized_week.csv / .png => result from running the default 

NOTE:
- The formatting of the real operation csv was for compatibility with the JKM simulation, which can read this as an input that defines the operating state of each hour. 
- The Aug week starts on 08/13/2021
- The October week starts on 10/08/2021