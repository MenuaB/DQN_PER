class LinearScheduler:
    def __init__(self, schedule_time_steps, initial_value=1.0, final_value=0.1):
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_time_steps = schedule_time_steps

    def get_value(self, time_step):
        fraction = min(float(time_step) / self.schedule_time_steps, 1)
        return self.initial_value + fraction * (self.final_value - self.initial_value)


if __name__ == "__main__":
    ls = LinearScheduler(schedule_time_steps=100, initial_value=1.0, final_value=0.1)
    for i in range(100):
        print(ls.get_value(i))