from concurrent.futures import process
from sat_controller import SatControllerInterface, sat_msgs

import numpy as np

# Team code is written as an implementation of various methods
# within the the generic SatControllerInterface class.
# If you would like to see how this class works, look in sat_control/sat_controller

# Specifically, init, run, and reset

class TeamController(SatControllerInterface):
    """ Team control code """

    def team_init(self):
        """ Runs any team based initialization """
        # Run any initialization you need
        self.max_thrust_force = 0.5  # Newtons

        self.dt = 0.05  # seconds

        self.kp = 5.0
        self.ki = 0.0
        self.kd = 2.0

        # Example of persistant data
        self.counter = 0
        self.errors = np.array([0, 0, 0], dtype=float)
        self.errors_previous = np.array([0, 0, 0], dtype=float)
        self.errors_integral = np.array([0, 0, 0], dtype=float)

        # Example of logging
        self.logger.info("Initialized :)")
        self.logger.warning("Warning...")
        self.logger.error("Error!")

        # Update team info
        team_info = sat_msgs.TeamInfo()
        team_info.teamName = "Team 17"
        team_info.teamID = 17

        # Return team info
        return team_info

    def team_run(self, system_state: sat_msgs.SystemState, satellite_state: sat_msgs.SatelliteState, dead_sat_state: sat_msgs.SatelliteState) -> sat_msgs.ControlMessage:
        """ Takes in a system state, satellite state """

        print("Dead Sat State:")
        print(dead_sat_state)

        print("Live Sat State:")
        print(satellite_state)

        current_position = np.array([satellite_state.pose.x, satellite_state.pose.y, satellite_state.pose.theta], dtype=float)
        desired_position = np.array([dead_sat_state.pose.x, dead_sat_state.pose.y, dead_sat_state.pose.theta], dtype=float)

        # Get timedelta from elapsed time
        elapsed_time = system_state.elapsedTime.ToTimedelta()
        self.logger.info(f'Elapsed time: {elapsed_time}')

        # Example of persistant data
        self.counter += 1
        (errors, errors_integral, errors_derivative) = self.error_calc(desired_position, current_position)

        # Example of logging
        self.logger.info(f'Counter value: {self.counter}')

        # PID control
        process_variable = (self.kp * errors) + (self.ki * errors_integral) + (self.kd * errors_derivative)
        print("Process Variable: ")
        print(process_variable)

        thrust_force = self.thrust_force_calcs(process_variable)

        # Create a thrust command message
        control_message = sat_msgs.ControlMessage()

        # Set thrust command values, basic PD controller that drives the sat to [0, -1]
        # control_message.thrust.f_x = -2.0 * (satellite_state.pose.x - (dead_sat_state.pose.x - 0.2)) - 6.0 * satellite_state.twist.v_x
        # control_message.thrust.f_y = -2.0 * (satellite_state.pose.y - (dead_sat_state.pose.y)) - 6.0 * satellite_state.twist.v_y
        # control_message.thrust.tau = -2.0 * (satellite_state.pose.theta - (dead_sat_state.pose.theta+3.1415)) - 6.0 * satellite_state.twist.omega

        (control_message.thrust.f_x, control_message.thrust.f_y, control_message.thrust.tau) = thrust_force
        print("Thrust: ")
        print(thrust_force)

        # Return control message
        return control_message

    def team_reset(self) -> None:
        # Run any reset code
        pass

    def error_calc(self, desired_position, current_position):
        """Calculates the system errors, given the desired and current positions"""

        self.errors = desired_position - current_position
        self.errors_integral += self.errors * self.dt
        self.errors_derivative = (self.errors - self.errors_previous) / self.dt
        self.errors_previous = self.errors

        print("Errors: ")
        print(self.errors)


        return self.errors, self.errors_integral, self.errors_derivative
    
    def modified_sigmoid(self, x):
        return 2.0 * (1.0 / (1.0 + np.exp(-x))) - 1.0
    
    def thrust_force_calcs(self, process_variable):

        thrust_percents = np.ndarray(len(self.errors))

        for variable in range(0, len(thrust_percents)):
            if process_variable[variable] < 1:
                thrust_percents[variable] = -1.0 * round(self.modified_sigmoid(-process_variable[variable]), 2)
            elif process_variable[variable] > 1:
                thrust_percents[variable] = round(self.modified_sigmoid(process_variable[variable]), 2)

        thrust_force = thrust_percents * self.max_thrust_force

        return thrust_force
