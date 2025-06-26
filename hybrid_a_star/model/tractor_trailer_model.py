from model.base_model import BaseModel
import casadi as cs
import casadi  # only for type hinting
from typing import Union

class TractorTrailerModel(BaseModel):
    """!
    @brief Model of a tractor-trailer system.

    This class extends BaseModel and defines the dynamics of a tractor-trailer system 
    with different configurations depending on the number of state variables.
    """
    def __init__(self, params: dict) -> None:
        """!
        @brief Initializes the Tractor-Trailer Model.

        @param[in] params Dictionary containing model parameters:
                         - "length_back" (float): Length from the center of the tractor to the hitching point.
                         - "length_front" (float): Length from the center of the trailer to the hitching point.
                         - "trailer_based_model" (bool): Flag to determine the model configuration. True if the model is trailer-based, False otherwise.
        """
        super().__init__(params)

        ## Length from the center of the tractor to the hitching point
        self._lb = params["length_back"]
        
        ## Length from the center of the trailer to the hitching point
        self._lf = params["length_front"]
        
        ## Flag to determine the model configuration. True if the model is trailer-based, False otherwise. Default is False.
        self._trailer_based_model = params.get("trailer_based_model", False)
        
        print(f"Tractor-Trailer Model was successfully initialized with {self.nx} states and {self.nu} inputs")
        
        if self._trailer_based_model:
            print(f"Tractor-Trailer Model is trailer-based")
        else:
            print(f"Tractor-Trailer Model is tractor-based")
    
    def dynamics(self, state: Union[casadi.SX, casadi.DM], input: Union[casadi.SX, casadi.DM]) -> Union[casadi.SX, casadi.DM]:
        """!
        @brief Computes the system dynamics for the tractor-trailer model.

        @param[in] state A CasADi SX or DM variable representing the state of the system.
        @param[in] input A CasADi SX or DM variable representing the control input.
        
        @return Union[casadi.SX, casadi.DM] A CasADi SX or DM variable representing the time derivative of the state.
        """
        if state.size() != (self.nx, 1):
            raise Exception(f"Failed to compute dynamics. The size of state {state.size()} is not matched with the required size ({self.nx}, 1)")
        
        if input.size() != (self.nu, 1):
            raise Exception(f"Failed to compute dynamics. The size of input {input.size()} is not matched with the required size ({self.nu}, 1)")
        
        v1 = input[0]
        
        w1 = input[1]
        
        state_dot = []
        
        if state.size1() == 6:
            x1, y1, theta1, x2, y2, theta2 = cs.vertsplit(state)
            
            gamma = theta2 - theta1
            
            x1_dot = v1 * cs.cos(theta1)
            
            y1_dot = v1 * cs.sin(theta1)
            
            theta1_dot = w1 
            
            x2_dot = v1 * cs.cos(theta2) * cs.cos(gamma) - w1 * self._lb * cs.cos(theta2) * cs.sin(gamma)
            
            y2_dot = v1 * cs.sin(theta2) * cs.cos(gamma) - w1 * self._lb * cs.sin(theta2) * cs.sin(gamma)
            
            theta2_dot = - v1 * (1 / self._lf) * cs.sin(gamma) - w1 * (self._lb / self._lf) * cs.cos(gamma)
            
            state_dot = [x1_dot, y1_dot, theta1_dot, x2_dot, y2_dot, theta2_dot]         
            
        elif state.size1() == 4 and self._trailer_based_model:
            x2, y2, theta2, gamma = cs.vertsplit(state)
            
            x2_dot = v1 * cs.cos(theta2) * cs.cos(gamma) - w1 * self._lb * cs.cos(theta2) * cs.sin(gamma)
            
            y2_dot = v1 * cs.sin(theta2) * cs.cos(gamma) - w1 * self._lb * cs.sin(theta2) * cs.sin(gamma)
            
            theta2_dot = - v1 * (1 / self._lf) * cs.sin(gamma) - w1 * (self._lb / self._lf) * cs.cos(gamma)
            
            gamma_dot = - v1 * (1 / self._lf) * cs.sin(gamma) - w1 * ((self._lb / self._lf) * cs.cos(gamma) + 1)
            
            state_dot = [x2_dot, y2_dot, theta2_dot, gamma_dot]
            
        elif state.size1() == 4 and not self._trailer_based_model:
            x1, y1, theta1, gamma = cs.vertsplit(state)
            
            x1_dot = v1 * cs.cos(theta1)
            
            y1_dot = v1 * cs.sin(theta1)
            
            theta1_dot = w1 
            
            gamma_dot = - v1 * (1 / self._lf) * cs.sin(gamma) - w1 * ((self._lb / self._lf) * cs.cos(gamma) + 1)
            
            state_dot = [x1_dot, y1_dot, theta1_dot, gamma_dot]
        
        return cs.vertcat(*state_dot)

    def compute_tractor_pose_from_trailer_pose(self, state: Union[casadi.SX, casadi.DM]) -> Union[casadi.SX, casadi.DM]:
        """!
        @brief Computes the pose of the tractor given the system state.

        @param[in] state A CasADi SX or DM variable representing the current state of the system (x2, y2, theta2, gamma).
        
        @return Union[casadi.SX, casadi.DM] A CasADi SX or DM variable representing the tractor pose (x1, y1, theta1).
        """
        if state.size() != (self.nx, 1):
            raise Exception(f"Failed to compute tractor pose. The size of input argument {state.size()} is not matched with the size of state ({self.nx}, 1)")

        x2, y2, theta2, gamma = cs.vertsplit(state)
        
        theta1 = theta2 - gamma
        
        x1 = x2 + self._lf * cs.cos(theta2) + self._lb * cs.cos(theta1)
        
        y1 = y2 + self._lf * cs.sin(theta2) + self._lb * cs.sin(theta1)
        
        return cs.vertcat(x1, y1, theta1)
    
    def compute_trailer_pose_from_tractor_pose(self, state: Union[casadi.SX, casadi.DM]) -> Union[casadi.SX, casadi.DM]:
        """!
        @brief Computes the pose of the trailer given the system state.

        @param[in] state A CasADi SX or DM variable representing the current state of the system (x1, y1, theta1, gamma).
        
        @return Union[casadi.SX, casadi.DM] A CasADi SX or DM variable representing the trailer pose (x2, y2, theta2).
        """
        if state.size() != (self.nx, 1):
            raise Exception(f"Failed to compute trailer pose. The size of input argument {state.size()} is not matched with the size of state ({self.nx}, 1)")
        
        x1, y1, theta1, gamma = cs.vertsplit(state)
        
        theta2 = theta1 + gamma
        
        x2 = x1 - self._lf * cs.cos(theta1) - self._lb * cs.cos(theta2)
        
        y2 = y1 - self._lf * cs.sin(theta1) - self._lb * cs.sin(theta2)
        
        return cs.vertcat(x2, y2, theta2)
    
    def get_full_state_trajectory(self, state_trajectory: Union[casadi.SX, casadi.DM]) -> Union[casadi.SX, casadi.DM]:
        """!
        @brief Gets the full state trajectory (tractor and trailer).
        @param[in] state_trajectory The state trajectory of either tractor or trailer (x,y,theta,gamma).
        @return Union[casadi.SX, casadi.DM] The full state trajectory (tractor and trailer) (x1,y1,theta1,x2,y2,theta2).
        """ 
        if state_trajectory.shape[0] != self.nx:
            raise Exception(f"Failed to compute full state trajectory. The size of input argument {state_trajectory.shape[0]} is not matched with the size of state {self.nx}")
        
        if type(state_trajectory) == casadi.SX:
            full_state_trajectory = casadi.SX.zeros((6, state_trajectory.shape[1]))
        else:
            full_state_trajectory = casadi.DM.zeros((6, state_trajectory.shape[1]))
        
        if state_trajectory.shape[0] == 6:
            print("The state trajectory is already the full state trajectory")
            return state_trajectory
        
        elif state_trajectory.shape[0] == 4 and self._trailer_based_model == False:
            full_state_trajectory[0:3, :] = state_trajectory[0:3, :] # tractor state    
            for i in range(state_trajectory.shape[1]):
                full_state_trajectory[3:6, i] = self.compute_trailer_pose_from_tractor_pose(state_trajectory[:, i])
                
        elif state_trajectory.shape[0] == 4 and self._trailer_based_model == True:
            full_state_trajectory[3:6, :] = state_trajectory[0:3, :] # trailer state
            for i in range(state_trajectory.shape[1]):
                full_state_trajectory[0:3, i] = self.compute_tractor_pose_from_trailer_pose(state_trajectory[:, i])
            
        return full_state_trajectory