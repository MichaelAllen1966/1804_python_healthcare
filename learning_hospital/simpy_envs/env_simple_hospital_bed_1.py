import random
import simpy

class HospGym:
    """
    A simple SimPy hospital simulation with an OpenAI gym-like interface for 
    Reinforcement Learning.
    
    Any environment needs:
    * A reward structure
    * An initialise (reset) method that returns the initial observations
    * A choice of actions 
    * A state space
    * A way to make sure the action is legal/possible
    * A way for the action to affect environment
    * A step function that returns the new observations, reward,
        whether state is terminal, additional information
    * A way to render the environment.
    * A way to recognise and return a terminal state (end of episode)
    
    
    Internal methods:
    -----------------

    __init__:
        Constructor method.
    _adjust_bed_numbers:
        Introduces delay before bed numbers actually change
    _adjust_pending_bed_change:
        Track pending bed changes in state dictionary
    _calculate_reward:
        Calculates reward based on empty beds or beds without patient
    _load_patients:
        Inital load of patients into hospital (avoid starting empty)
    _new_admission:
        Loop creating new patient admissions
    _patient_spell:
        Patient spell in hospital sim
        
        
    Interfacing methods:
    --------------------
    render:
        Display state 
    reset:
        Initialise environment
        Return first state observations
    step:
        Take an action. Update state. Return obs, reward, terminal, info
        
    
    Input parameters (converted to attributes)
    ------------------------------------------
        
    arrivals_per_day:
        Average arrivsls per day
    delay_to_change_beds:
        Time between requesting change in beds, and change in beds happening
    los:
        Average patient length of stay
    penalty_for_empty_bed:
        reward penalty for each empty beds
    penalty_for_patients_without_bed:
        reward penalty for  each patient without bed
    render_env:
        Boolean, render state each action?
    sim_duration:
        Length of simulation run (days)  
        
    
    Additional attributes:
    ----------------------    

    action_size:
        Number of possible actions   
    actions:
        List of possible actions
    arrivals_by_day:
        Dictionary of average arrivals by day of week
    observation_size:
        Number of features in observation space
    state:
        SimPyGymState object (dictionary)
        
        
    State dictionary
    ----------------
    
    The state dictionary contains the following items:
        weekday: day of week (0-6)
        beds: number of available beds (free or occupied)
        patients: number of patients in hospital
        spare_beds: number of beds without patient
        pending_bed_change: pending requests for bed changes
            

    """
    
    def __init__(self, arrivals_per_day=100, delay_to_change_beds=2, los=5,
                 penalty_for_empty_bed=1, penalty_for_patients_without_bed=1.25,
                 render_env=False, sim_duration=365):
                 
        """
        Constructor method for HospGym class.
        
        Input Parameters
        ----------------
        
        arrivals_per_day:
            Average arrivsls per day
        delay_to_change_beds:
            Time between requesting change in beds, and change in beds happening
        los:
            Average patient length of stay
        penalty_for_empty_bed:
            reward penalty for each empty beds
        penalty_for_patients_without_bed:
            reward penalty for  each patient without bed
        render_env:
            Boolean, render simulation
        sim_duration:
            Length of simulation run (days)        
        """
        
        # set average length of stay
        self.los = los
        
        # Set average arrivals per day
        self.arrivals_per_day = arrivals_per_day
        
        # Set up state dictionary 
        self.state = dict()
        # Weekday count (used for periodicity of demand)
        self.state['weekday'] = 0
        # Number of beds in hospital
        self.state['beds'] = 0
        # Number of patients in hospital
        self.state['patients'] = 0
        # Number of spare beds
        self.state['spare_beds'] = 0
        # Tally of bed adjustments waiting
        self.state['pending_bed_change'] = 0
        # Show environemnt on each action?
        self.render_env = render_env
        
        # Action space = change in number of beds: -20 to +20 in a day
        # 5 actions for change bed number by -20, -10, 0, +10, +20
        # Bed numbers change after set delay
        self.actions = [0,1,2,3,4]
        
        # Set delay_to_change_beds
        self.delay_to_change_beds = delay_to_change_beds
        
        # Set up costs
        self.penalty_for_empty_bed = penalty_for_empty_bed
        self.penalty_for_patients_without_bed = penalty_for_patients_without_bed
        
        # Set sim duration (returns Terminal state after this)
        self.sim_duration = sim_duration
        
        # Set up observation and action space sizes
        self.observation_size = 5
        self.action_size = 5
        
        # Set up dictionary of arrivals by day of week
        # Weekend days are 50%, and weekday days are 120% of average arrivals 
        self.arrivals_by_day = dict()
        for day_num in range(7):
            if day_num < 5:
                self.arrivals_by_day[day_num] = arrivals_per_day * 1.2
            else:
                self.arrivals_by_day[day_num] = arrivals_per_day * 0.5       
  
    
    def _adjust_bed_numbers(self, action):       
        
        """
        introduces a delay before bed numbers actually change.
        If delay >0 then reduce by 0.001 to include count in next action return
        """
        
        self.state['spare_beds'] = self.state['beds'] - self.state['patients']
        
        # Simpy environemnt timeout (delay) before bed numbers change
        delay = max(self.delay_to_change_beds - 0.001, 0)
        yield self.env.timeout(delay)
        
        # Update state weekday
        self.state['weekday'] = int((self.env.now) % 7)   
        
        # Adjust beds
        if action == 0:
            self.state['beds'] -= 20
            self.state['pending_bed_change'] += 20
        elif action == 1:
            self.state['beds'] -= 10
            self.state['pending_bed_change'] += 10
        elif action == 3:
            self.state['beds'] += 10
            self.state['pending_bed_change'] -= 10
        elif action == 4:
            self.state['beds'] += 20
            self.state['pending_bed_change'] -= 20
            
        self.state['spare_beds'] = self.state['beds'] - self.state['patients']
            
            
    def _adjust_pending_bed_change(self, action):
        """
        Adjust tracker (in state dictionary) of bed changes requested but not
        yet carried out.
        """
                
        # Update state weekday
        self.state['weekday'] = int((self.env.now) % 7)
        
        # Adjust pending bed changes
        if action == 0:
            self.state['pending_bed_change'] -= 20
        elif action == 1:
            self.state['pending_bed_change'] -= 10
        elif action == 3:
            self.state['pending_bed_change'] += 10
        elif action == 4:
            self.state['pending_bed_change'] += 20        
            
    
    def _calculate_reward(self):
        """
        Calculate reward (always negative or 0)
        """
                       
        if self.state['spare_beds'] > 0:
            loss = - (self.state['spare_beds'] * self.penalty_for_empty_bed)
        elif self.state['spare_beds'] < 0:
            loss = (self.state['spare_beds'] * 
                    self.penalty_for_patients_without_bed)
        else:
            loss = 0
                    
        return loss
    
    def _load_patients(self):
        """
        Load hospital accoriding to calulated average occupancy. Assume average
        los of patients load = half of total avaerage los
        """
        
        number_to_load = self.arrivals_per_day * self.los
        for patient in range(number_to_load):
            self.state['beds'] += 1
            self.state['patients'] += 1
            self.env.process(self._patient_spell(los_adjustment=0.5))
            
    
    def _new_admission(self):
        """
        New admissions to hospital.
        Sample inter-arrival times from inverse exponential distribution.
        Inter-arrival times depend on day of week.
        """
        while True:
            # Adjust hospital patient counts
            self.state['patients'] += 1
            self.state['spare_beds'] = (
                self.state['beds'] - self.state['patients'])
        
            # Call patient spell process
            self.env.process(self._patient_spell())
            
            # Update weekday
            self.state['weekday'] = int((self.env.now) % 7)

            # Set and call delay before looping back to new patient admission
            interarrival_time = 1 / self.arrivals_by_day[self.state['weekday']]
            next_admission = random.expovariate(1 / interarrival_time)
            yield self.env.timeout(next_admission)
            
            
    def _patient_spell(self, los_adjustment=1):
        """
        Patient spell in hospital. 
        Sample length of stay from inverse exponential distribution.
        los_adjustment is used to adjust average length of stay for patients
          loaded into model at start (assume remaining los is half normal los)
        """
        
        # Get length of stay from distributin
        patient_los = random.expovariate(1 / (self.los * los_adjustment))
        
        # Simulation timeout for length of stay
        yield self.env.timeout(patient_los)
        
        # Update weekday
        self.state['weekday'] = int((self.env.now) % 7)
            
        # Adjust patient and bed counts as patient leaves hospital
        self.state['patients'] -= 1
        self.state['spare_beds'] = self.state['beds'] - self.state['patients']
       
    
    def render(self):
        """Display current state"""
        
        print (f"Weekday: {self.state['weekday']}, ", end = '')
        print (f"Beds: {self.state['beds']}, ", end = '')
        print (f"Patients: {self.state['patients']}, ", end = '')
        print (f"Spare beds: {self.state['spare_beds']}, ", end = '')
        print (f"Pending bed change: {self.state['pending_bed_change']}")
        
    
    def reset(self):
        """Reset environemnt"""
        
        # Initialise simpy environemnt
        self.env = simpy.Environment()
        
        # Set up starting processes
        self.env.process(self._new_admission())
        
        # Set starting state values
        self.state['weekday'] = 0
        self.state['beds'] = 0
        self.state['patients'] = 0
        self.state['spare_beds'] = 0
        self.state['pending_bed_change'] = 0
        
        # Inital load of patients (to average occupancy)
        self._load_patients()
        
        # Put state dictionary items into observations list
        observations = [v for k,v in self.state.items()]
        
        # Return starting state observations
        return observations
        

    def step(self, action):
        
        """
        Interaction with environemnt. Actions are:
            0: Request beds to be reduced by 10
            1: Request beds to be reduced by 5
            2: No change in beds requested
            3: Request beds to be increased by 5
            4: Request beds to be increased by 10
            
        There is a delay between request for bed number change and the number of
        changes actually occuring (specified in self.delay_to_change_beds).
        
        The step method:
        1. Tracks changes to requested bed numbers
        2. Updates weekday
        3. Calls bed change process
        4. Puts state dictionary items into observations list
        5. Checks whether terminal state reached (based on sim time)
        6. Get reward
        7. Creates empty info dictionary (used to be compatble with OpenAI Gym)
        8. Renders environemnt if requested
        9. Returns (observations, reward, terminal, info)
                
        Returns
        -------
        * observations: weekday, beds, patients, spare_beds, pending_bed_change
        * reward: -1 for each unoccupied bed, -3 for each patient without bed
        * terminal: if sim has reached specified duration
        * info: an empty dictionary
            
        """
        
        # Adjust pending bed change (tracks changes in beds due)
        self._adjust_pending_bed_change(action)
        
        # Update state weekday
        self.state['weekday'] = int((self.env.now) % 7)
        
         # Call bed change process
        self.env.process(self._adjust_bed_numbers(action))       
        
        # Put state dictionary items into observations list
        observations = [v for k,v in self.state.items()]
        
        # Check whether terminal state reached (based on sim time)
        terminal = True if self.env.now >= self.sim_duration else False
        
        # Get reward
        reward = self._calculate_reward()
        
        # Information is empty dictionary (used to be compatble with OpenAI Gym)
        info = dict()
        
        # Render environment if requested
        if self.render_env:
            self.render()
        
        # Return tuple of observations, reward, terminal, info
        return (observations, reward, terminal, info)
