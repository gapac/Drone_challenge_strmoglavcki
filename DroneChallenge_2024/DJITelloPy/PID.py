class PIDRegulator:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.prev_error = 0

    def calculate(self, setpoint, current_value):
        error = setpoint - current_value

        # Proportional term
        p = self.kp * error

        # Integral term
        self.error_sum += error
        i = self.ki * self.error_sum

        # Derivative term
        d = self.kd * (error - self.prev_error)
        self.prev_error = error

        # Calculate the output
        output = p + i + d

        return output