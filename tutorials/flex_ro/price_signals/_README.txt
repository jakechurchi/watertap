NAMING CONVENTION:
- season => summer or winter where "winter" is October - May
- Time length => day = 24 hrs, week = 168 hrs, month = Depends on the month, but assumed 30 days by default
- suffix => name of the rate structure
    - No suffix => The plant's rate structure from 2023. This included TOU3 + some modifications from their subscription through PRIME. The values are pulled from their invoices. 
    - TOU8 => Southern California Electric Time of Use 8 rate from 2026
    - CPP => Critical Peak Pricing * There are some caveats. Combined with the their 2023 existing rate.
    - ELRP => Emergency Load Reduction Program ($2 / kWh reduced). Combined with their 2023 existing rate.
    - 2026 => TOU3 rate from 2026
    - flat_escalation => Atr
    - RTP => Real Time Pricing. See more info:
        - low_RTP => Low Winter weekday + low weekend. This is the lowest possible electricity price week under this rate structure
        - hot_RTP => hot summer weekday + high weekend. This is the highest possible electricity price week under this rate structure

NOTES:
- All price signals are hourly, though pricetaker is compatable with shorter time steps. 
- The different rate structures are from different years.