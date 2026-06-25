
Flexible RO No 2
============================

Introduction
------------

This flowsheet represents the operational window and constraints of a water treatment plant. It was developed using operational data from the Water Replenishment District (WRD) ARC facility in Pico Rivera, CA.The plant is modeled by 4 RO trains, 3 UF pumps and 1 UV unit. The energy intensity of each RO train is a function of flowrate and recovery. This is the key difference between theis implementation and flex_ro_1.

The flowsheet is used with the Pricetaker model to determine the cost-optimal operation based on the treatment system energy surrogates and constraints as well as the variable grid electricity costs.


Model Structure
---------------

The flowsheet is 

The parameters of each unit model are defined in params. They are passed during the model build.


Unit Models
-----------

Pretreatment (UF)
~~~~~~~~~~~~~~~~~

The UF unit uses the ``WRD_UFParams`` dataclass in
``watertap.flowsheets.flex_desal.params``.

.. list-table:: WRD UF parameters
   :header-rows: 1

   * - Parameter
     - Default value
     - Units
     - Description
   * - ``num_uf_pumps``
     - ``4``
     - -
     - Number of UF pumps represented in the model.
   * - ``minimum_operating_pumps``
     - ``1``
     - -
     - Minimum number of UF pumps that must operate.
   * - ``allow_shutdown``
     - ``True``
     - -
     - Enables UF unit on/off commitment logic.
   * - ``minimum_flowrate``
     - ``344``
     - m3/h
     - Minimum UF pump flowrate when operating (m3/h).
   * - ``nominal_flowrate``
     - ``900``
     - m3/h
     - Nominal UF pump flowrate (m3/h).
   * - ``maximum_flowrate``
     - ``989``
     - m3/h
     - Maximum UF pump flowrate (m3/h).
   * - ``nominal_recovery``
     - ``1``
     - dimensionless
     - Nominal UF recovery.
   * - ``minimum_uptime``
     - ``2``
     - time steps
     - Minimum number of time steps the UF pump stays on after startup.
   * - ``minimum_downtime``
     - ``2``
     - time steps
     - Minimum number of time steps the UF pump stays off after shutdown.
   * - ``startup_delay``
     - ``1``
     - time steps
     - Delay (time steps) between startup command and operation.
   * - ``allow_variable_recovery``
     - ``False``
     - -
     - Recovery is fixed at nominal value for this WRD UF model.
   * - ``surrogate_type``
     - ``quadratic_energy_intensity``
     - -
     - UF energy intensity surrogate form.
   * - ``surrogate_a``, ``surrogate_b``, ``surrogate_c``
     - ``1``, ``1``, ``1``
     - surrogate-dependent
     - Coefficients for the quadratic UF energy intensity surrogate.


RO
~~

The RO unit uses the ``WRD_ROParams`` dataclass in
``watertap.flowsheets.flex_desal.params``.

.. list-table:: WRD RO parameters
   :header-rows: 1

   * - Parameter
     - Default value
     - Units
     - Description
   * - ``num_ro_skids``
     - ``4``
     - -
     - Number of RO skids represented in the model.
   * - ``minimum_operating_skids``
     - ``2``
     - -
     - Minimum number of RO skids that must operate.
   * - ``allow_shutdown``
     - ``True``
     - -
     - Enables RO skid on/off commitment logic.
   * - ``minimum_flowrate``
     - ``0``
     - m3/h
     - Minimum RO skid flowrate when operating (m3/h).
   * - ``nominal_flowrate``
     - ``337.670``
     - m3/h
     - Nominal RO skid flowrate (m3/h).
   * - ``maximum_flowrate``
     - ``400``
     - m3/h
     - Maximum RO skid flowrate (m3/h).
   * - ``minimum_recovery``
     - ``0.88``
     - dimensionless
     - Minimum RO recovery.
   * - ``nominal_recovery``
     - ``0.92``
     - dimensionless
     - Nominal RO recovery.
   * - ``maximum_recovery``
     - ``0.925``
     - dimensionless
     - Maximum RO recovery.
   * - ``minimum_uptime``
     - ``2``
     - time steps
     - Minimum number of time steps a skid stays on after startup.
   * - ``minimum_downtime``
     - ``2``
     - time steps
     - Minimum number of time steps a skid stays off after shutdown.
   * - ``startup_delay``
     - ``1``
     - time steps
     - Delay (time steps) between startup command and operation.
   * - ``allow_variable_recovery``
     - ``False``
     - -
     - Recovery is bounded but not optimized as a free variable in default setup.
   * - ``surrogate_type``
     - ``constant_energy_intensity``
     - -
     - RO surrogate type selector for the WRD case.
   * - ``surrogate_file``
     - ``None``
     - path
     - Optional file path for a loaded RO energy surrogate.
   * - ``surrogate_a``, ``surrogate_b``, ``surrogate_c``
     - ``1``, ``1``, ``1``
     - surrogate-dependent
     - Coefficients used by the selected RO surrogate formulation.


Posttreatment (UV)
~~~~~~~~~~~~~~~~~~

The UV posttreatment unit uses the ``PosttreatmentParams`` dataclass in
``watertap.flowsheets.flex_desal.params``.

.. list-table:: WRD UV posttreatment parameters
   :header-rows: 1

   * - Parameter
     - Default value
     - Units
     - Description
   * - ``energy_intensity``
     - ``0.41``
     - kWh/m3
     - Specific UV energy intensity.
   * - ``leakage_fraction``
     - ``0``
     - dimensionless
     - Fraction of inlet flow not recovered in UV posttreatment.
   * - ``chemical_cost``
     - ``None``
     - $/m3
     - Optional variable chemical cost ($/m3).


Variables
---------



Equations and Relationships
---------------------------



Functions
-----------
In addition to the functions described in the first tutorial, the WRD implementation
uses several helper functions in
``watertap.flowsheets.flex_desal.wrd_ro_flowsheet``:

* ``add_flow_costs(m)``
  Adds total feed, brine-discharge, and chemical-cost expressions over the time horizon.

* ``add_flow_changes_penalty(m)``
  Introduces binary variables and Big-M constraints to detect RO/UF flowrate changes
  between time steps, then adds a penalty expression for frequent changes.

* ``calculate_replacement_costs(m)``
  Computes a flexibility metric from shutdown behavior and uses that metric to form
  annualized replacement-cost expressions for configured replacement categories. The equations are given below:



* ``calculate_flexibility_metrics(m, baseline_power, baseline_electricity_cost, baseline_replacement_cost)``
  Post-processes solved results to estimate flexibility quantities such as charge/discharge
  capacities and levelized value/cost of flexibility.

* ``begin_and_end_constraint(m)``
  Enforces cyclic operation by matching RO train 1 operating mode at the first and
  last time points.






Flowsheet Specifications
------------------------

The operational limits and costing values used in this flowsheet are based on the WRD plant.

[Insert table of limits for recovery and flowrates for UF and RO units]


References
----------