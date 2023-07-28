from numpy import round, random
from matplotlib import patches
from random import choice
from graph import Graph


class Robot:
    graph = Graph()

    # static member variable
    delT = 0.05
    jMax = 5
    vMax = 1.6
    aMax = 1.5
    vAvg = 1.2  # For reward calculation

    # For random initial condition
    # range of distance
    distanceRange = [1, 7]

    # Robot plotting parameter
    xSize = 0.25
    ySize = 0.25

    def __init__(self, target=None, v0=None, a0=None):
        self.state_size = 3  # goalDistance, velocity and acceleration
        self.action_size = 3  # Increment in acceleration, constant acceleration and decrement in acceleration

        self.target = target  # Goal target distance
        self.position = 0  # Position of the robot
        self.t = 0  # time
        self.goalReached = False

        # Robot 3 States
        self.goalDistance = self.target - self.position if self.target is not None else None
        self.velocity = v0
        self.acceleration = a0

        # Robot current Action
        self.action = None

        self.name = "Zippy_RL"
        self.robotColor = "#" + ''.join([choice('0123456789ABCDEF') for j in range(6)])

        self.patch = patches.Rectangle((0, 0), Robot.xSize, Robot.ySize, fc=self.robotColor,
                                       rotation_point='center', edgecolor='k')
        self.pathPatch = patches.Rectangle((0, 0), 0, 0, fc=self.robotColor,
                                           rotation_point='center', edgecolor='k')

    def step(self, action):
        if action == 0:
            # Increment in acceleration
            self.acceleration += Robot.jMax * Robot.delT
            self.acceleration = max(-Robot.aMax, min(Robot.aMax, self.acceleration))
            self.velocity += self.acceleration * Robot.delT
            # self.velocity = max(0.0, min(Robot.vMax, self.velocity))
            self.position += self.velocity * Robot.delT
            self.goalDistance = self.target - self.position
        elif action == 1:
            # No change in acceleration
            # self.acceleration = max(-Robot.aMax, min(Robot.aMax, self.acceleration))
            self.velocity += self.acceleration * Robot.delT
            # self.velocity = max(0.0, min(Robot.vMax, self.velocity))
            self.position += self.velocity * Robot.delT
            self.goalDistance = self.target - self.position
        elif action == 2:
            # Decrement in acceleration
            self.acceleration += -Robot.jMax * Robot.delT
            self.acceleration = max(-Robot.aMax, min(Robot.aMax, self.acceleration))
            self.velocity += self.acceleration * Robot.delT
            # self.velocity = max(0.0, min(Robot.vMax, self.velocity))
            self.position += self.velocity * Robot.delT
            self.goalDistance = self.target - self.position
        else:
            print("[Action] Type error")

        self.t += Robot.delT

        # Calculate reward function
        reward, rewardBreakups = self.calculate_reward()

        # Check goal reached
        doneFlag = False
        if self.goalDistance <= - self.target or self.goalDistance > 2 * self.target or self.t > 2 * self.target / Robot.vAvg:
            doneFlag = True

        return [self.goalDistance, self.velocity, self.acceleration], self.t, reward, doneFlag, rewardBreakups

    def reset(self, target=None, v0=None, a0=None):
        self.target = target if target is not None else round(
            random.uniform(Robot.distanceRange[0], Robot.distanceRange[1]), 2)

        self.goalReached = False
        self.t = 0

        # Robot 4 States
        self.position = 0
        self.goalDistance = self.target - self.position
        self.velocity = v0 if v0 is not None else random.uniform(0, Robot.vMax)
        self.acceleration = a0 if a0 is not None else random.uniform(-Robot.aMax, Robot.aMax)

        return [self.goalDistance, self.velocity, self.acceleration]

    def calculate_reward(self):
        rewardBreakups = dict(rGoalDistance=0,
                              rSlowerThanAvgTime=0,
                              rVLimitCrossing=0,
                              rALimitCrossing=0,
                              rUnnesAcc=0,
                              rMovingBack=0,
                              rMovingBeyondGoal=0,
                              rReachingGoal=0,
                              rReachingGoalWithAcc=0,
                              rReachingGoalWithVel=0)

        # Can still optimise the reward function
        # Need to check the percentage contribution of each reward segment by segment

        # Calculate the reward based on different criteria
        r = 0

        # Penalize for being far away from the goal
        if self.goalDistance < self.target:
            rewardBreakups["rGoalDistance"] = -10 * self.goalDistance
            r += rewardBreakups["rGoalDistance"]

        # Penalize for moving slower than average time
        if self.t > self.target / Robot.vAvg:
            rewardBreakups["rSlowerThanAvgTime"] = -10 * self.t
            r += rewardBreakups["rSlowerThanAvgTime"]

        # Penalize heavily for crossing vMax and vMin limit
        if self.velocity > Robot.vMax or self.velocity < 0:
            rewardBreakups["rVLimitCrossing"] = -250
            r += rewardBreakups["rVLimitCrossing"]

        # Penalize heavily for crossing aMax and aMin limit
        if self.acceleration > Robot.aMax or self.acceleration < -Robot.aMax:
            rewardBreakups["rALimitCrossing"] = -500
            r += rewardBreakups["rALimitCrossing"]

        # Penalize unnecessary use of acceleration
        if abs(self.acceleration) >= 0:
            rewardBreakups["rUnnesAcc"] = - 10 * abs(self.acceleration)
            r += rewardBreakups["rUnnesAcc"]

        # Penalize for moving backward
        if self.goalDistance > self.target:
            rewardBreakups["rMovingBack"] = -2000
            r += rewardBreakups["rMovingBack"]

        # Penalize heavily for moving beyond the goal
        if self.goalDistance < 0:
            rewardBreakups["rMovingBeyondGoal"] = -1000
            r += rewardBreakups["rMovingBeyondGoal"]

        # Rewards for reaching the goal
        if round(self.goalDistance, 1) == 0.0:
            # reaching goal reward
            rewardBreakups["rReachingGoal"] = 5000
            r += rewardBreakups["rReachingGoal"]

            # # Encourage maintaining desired velocity at the goal
            # r -= 0.1 * self.velocity
            #
            # # Encourage maintaining desired acceleration at the goal
            # r -= 0.1 * self.acceleration

            if round(self.acceleration, 1) == 0.0:
                rewardBreakups["rReachingGoalWithAcc"] = 5000
                r += rewardBreakups["rReachingGoalWithAcc"]

            if round(self.velocity, 1) == 0.0:
                rewardBreakups["rReachingGoalWithVel"] = 10000
                r += rewardBreakups["rReachingGoalWithVel"]

        return r, rewardBreakups
