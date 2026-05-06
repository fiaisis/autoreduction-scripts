# Vanadium
from isis_powder.gem import Gem

Gem_object = Gem(user_name="GEM_vanadium",
                            config_file=r"C:\Users\xhg73778\Documents\GEM\GEM_Mantid_Cycle_25_3\Cycle_25_3\Gem_config_example_25_3.yaml"
                            )
                            
Gem_object.create_vanadium(first_cycle_run_no=95070, # 97482,
                            mode="PDF",
                            do_absorb_corrections=True,
                            multiple_scattering=False,
                            spline_coefficient=120,
#                           texture_mode=True
                            )

# Focus

Gem_object = Gem(user_name="Breternitz", 
                            config_file='B:\Cycle_25_1\Gem_config_example_25_1.yaml'
                            )
                            

Gem_object.focus(run_number="97486",  # "1, 3, 5-7"
                            #unit_to_keep="dSpacing", # dSpacing, TOF
                            mode = "PDF",  # PDF, Rietveld
                            input_mode="Summed", # Summed, Individual
                            keep_raw_workspace=False
#                          ,vanadium_normalisation=False 
#,file_ext="s01"
#                           ,focused_cropping_values = # Values for Rietveld:
#                           [(700, 19500),  # Bank 1
#                           (1000, 19500),  # Bank 2
#                           (1000, 19500),  # Bank 3
#                           (1000, 19500),  # Bank 4
#                           (1000, 18500),  # Bank 5
#                           (1000, 16750)   # Bank 6
#                           ]
                            ,focused_cropping_values = # Values for PDF:
                            [(550, 19900),  # Bank 1
                            (550, 19900),  # Bank 2
                            (550, 19900),  # Bank 3
                            (550, 19900),  # Bank 4
                            (550, 18500),  # Bank 5
                            (550, 16750)   # Bank 6
                            ]
                            )