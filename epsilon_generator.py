import math

class EpsilonExponential:
    def __init__(self, st: float, ed: float, steps: int):
        self.st = st
        self.ed = ed
        self.steps = steps
        self.current_step = 0
        self.factor = (math.log10(ed) - math.log10(st)) / (steps - 1)
        
    def as_list(self):
        return [self.st * math.pow(10, self.factor * i) for i in range(self.steps)]
    
class EpsilonLinear:
    def __init__(self, st: float, ed: float, steps: int):
        self.st = st
        self.ed = ed
        self.steps = steps
        self.current_step = 0
        self.increment = (ed - st) / (steps - 1)
        
    def as_list(self):
        return [self.st + self.increment * i for i in range(self.steps)]
    
class EpsilonFixed:
    def __init__(self, value: float, steps: int):
        self.value = value
        self.steps = steps
        
    def as_list(self):
        return [self.value for _ in range(self.steps)]
    
class EpsilonGenerator:
    def __init__(self, schedules: list, final_value: float):
        self.schedules = schedules
        self.final_value = final_value
        self.epsilon_values = self._generate_epsilon_values()
        
    def _generate_epsilon_values(self):
        epsilon_values = []
        for schedule in self.schedules:
            epsilon_values.extend(schedule.as_list())
        
        return epsilon_values
    
    def get_nth_epsilon(self, n: int):
        if n < len(self.epsilon_values):
            return self.epsilon_values[n]
        else:
            return self.final_value
    
if __name__ == "__main__":
    ...