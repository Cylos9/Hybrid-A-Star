from abc import ABC, abstractmethod
import casadi #only for type hinting
import casadi as cs
from typing import Union

class BaseModel(ABC):
    """!
    @brief Abstract base class for dynamic system models.

    This class defines a template for dynamic systems, requiring derived classes 
    to implement the `dynamics` function. It also provides a numerical integration 
    method for advancing system states.
    """
    def __init__(self, params: dict) -> None:
        """!
        @brief Initializes the model with system parameters.

        @param[in] params Dictionary containing system parameters:
                         - "num_states" (int): The number of states in the system.
                         - "num_inputs" (int): The number of control inputs.
        """
        super().__init__()
        ## Number of states
        self.nx = params["num_states"]
        ## Number of inputs
        self.nu = params["num_inputs"] 
        
    @abstractmethod
    def dynamics(self, state: Union[casadi.SX, casadi.DM], input: Union[casadi.SX, casadi.DM]) -> Union[casadi.SX, casadi.DM]:
        """!
        @brief Computes the time derivative of the state.

        This method must be implemented in derived classes.

        @param[in] state A CasADi SX or DM variable representing the current state of the system.
        @param[in] input A CasADi SX or DM variable representing the control input to the system.

        @return Union[casadi.SX, casadi.DM] A CasADi SX or DM variable representing the time derivative of the state.
        """
        pass

    def step(self, state: Union[casadi.SX, casadi.DM], input: Union[casadi.SX, casadi.DM], step_size: float, method: str = "RK4") -> Union[casadi.SX, casadi.DM]:
        """!
        @brief Advances the state of the system using a numerical integration method.

        This function integrates the system's equations using various discrete methods.

        @param[in] state A CasADi SX or DM variable representing the current state of the system.
        @param[in] input A CasADi SX or DM variable representing the control input to the system.
        @param[in] step_size The step size for numerical integration.
        @param[in] method The integration method to use. Supported methods:
                  - "RK1" (Euler method)
                  - "RK4" (Runge-Kutta 4th order)

        @throws Exception If the specified integration method is not supported.

        @return Union[casadi.SX, casadi.DM] A CasADi SX or DM variable representing the state of the system at the next time step.
        """
        
        if method == "RK1":
            state_dot = self.dynamics(state, input)
            state += step_size * state_dot

        elif method == "RK4":
            k1 = self.dynamics(state, input)
            k2 = self.dynamics(state + 1/2 * step_size * k1, input)
            k3 = self.dynamics(state + 1/2 * step_size * k2, input)
            k4 = self.dynamics(state + step_size * k3, input)

            state += 1/6 * step_size * (k1 + 2*k2 + 2*k3 + k4)
            
        else:
            raise Exception(f"The discrete method '{method}' is not supported")

        return state
