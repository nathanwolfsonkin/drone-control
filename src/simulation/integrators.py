class Integrator:
    """Integrator for a system of first-order ordinary differential equations
    of the form \dot x = f(t, x, u).
    """
    def __init__(self, dt, f):
        self.dt = dt
        self.f = f

    def step(self, t, x, u):
        raise NotImplementedError

class Euler(Integrator):
    def step(self, t, x, u):
        return x + self.dt * self.f(t, x, u)

class Heun(Integrator):
    def step(self, t, x, u):
        intg = Euler(self.dt, self.f)
        xe = intg.step(t, x, u) # Euler predictor step
        return x + 0.5*self.dt * (self.f(t, x, u) + self.f(t+self.dt, xe, u))

class RungaKutta4(Integrator):
   def step(self, t, x, u):
       x1 = self.dt * self.f(t, x, u)
       x2 = self.dt * self.f(t + 0.5*self.dt, x + 0.5*x1, u)
       x3 = self.dt * self.f(t + 0.5*self.dt, x + 0.5*x2, u)
       x4 = self.dt * self.f(t + 0.5*self.dt, x + x3, u)
       return x + (1/6)*(x1+2*x2+2*x3+x4)