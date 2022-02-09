For running the code it is required to install do-mpc (possible via pip) and tensorflow.

The considered system is similar to the one described in "./info_about_system/Deep_learning-based_embedded_mixed-integer_model_predictive_control.pdf".

To compare the performance of the MPC controller and a learned neural network controller run "./comparison_main.py".

Real weather data from a weather station at RUB is given in the folder data. Since the data is impure, some cleaning was done in "./data/gen_weather_files.py" and a resulting clean data set is given in "./data/exttemp_and_solrad_2008.pkl" spanning the first 80 days of 2008.

The goal of the controller is to maximize the amount of energy sold to the grid (P_grid) while not violating the constraints. The exact constraints can be read from the file "./template_mpc.py".

If there are further questions, please contact me at benjamin.karg@tu-dortmund.de and I will be happy to answer or give you an introduction to the code.

Best,
Benjamin
# xai4mpc
