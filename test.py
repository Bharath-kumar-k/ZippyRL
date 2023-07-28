import numpy as np
from math import pi
from matplotlib import animation, pyplot as plt
from matplotlib.animation import FuncAnimation


class test_agent():
    def __init__(self, env, agent, verbose, fileName, saveTestAnimation=False, **kw):
        self.env = env
        self.agent = agent
        self.verbose = verbose

        self.state = np.reshape(env.reset(), [1, self.env.state_size])
        self.action = 1  # Zero Jerk Action at the start

        self.agent.load(fileName)
        self.done = False

        self.t = 0  # Time step # counter

        self.timeVec = []
        self.distToGoalVec = []
        self.velVec = []
        self.accVec = []
        self.jerkVec = []
        # self.actionVec = []

        self.fig, self.axAnim, self.axDisToGo, self.axVec, self.axAcc, self.axAct, self.time_text, self.acc_text, self.vel_text, self.disToGo_text = self.env.graph.plot_graph()

        self.fig.suptitle("Zippy Testing", fontsize=20)

        self.jerkActionMap = [self.env.jMax, 0, -self.env.jMax]

        self.pathLine = None
        # self.pathLine, = self.ax.plot([], [], lw=2)

        # self.ani = animation.FuncAnimation(fig=self.fig, func=self.animate,
        #                                    init_func=self.init, frames=2500, interval=1, blit=True, repeat=False)
        self.saveTestAnimation = saveTestAnimation

    def init(self):
        self.axAnim.add_patch(self.env.patch)
        self.pathLine, = self.axAnim.plot([], [], lw=2)
        return []

    def animate_robot(self, robot_patch, x_, y_=0, theta_=0):
        robot_patch.set_xy([x_ - self.env.xSize / 2, y_ - self.env.ySize / 2])
        robot_patch.set_angle(theta_ * 180 / pi)

    def animate(self):
        # Create the animation using FuncAnimation
        anim = FuncAnimation(self.fig, self.update, frames=2500,
                             init_func=self.init, interval=1, blit=True, repeat=False)
        if self.done:
            anim.event_source.stop()

        if self.saveTestAnimation:
            FFwriter = animation.FFMpegWriter(fps=10)
            anim.save('TestAnimation', writer=FFwriter)

        # Show the animation
        plt.show()

    def update(self, _):

        Tf = 2 * self.env.target / self.env.vAvg

        action = self.agent.actual_act(self.state)
        # Perform the action in the environment
        next_state, time, reward, self.done, _ = self.env.step(action)
        next_state = np.reshape(next_state, [1, self.env.state_size])

        self.state = next_state
        self.t += 1

        x_node_pos, y_node_pos = self.env.graph.nodes[11].nodePos(return_with_node_id=False)
        x_robot_pos = x_node_pos + self.env.position
        y_robot_pos = y_node_pos + 0
        locGoalPos = [x_node_pos + self.env.target, y_node_pos + 0]

        self.animate_robot(self.env.patch, x_robot_pos, y_robot_pos, 0)
        self.pathLine.set_data([x_robot_pos, locGoalPos[0]], [y_robot_pos, locGoalPos[1]])

        self.timeVec.append(np.round(time, 2))
        self.distToGoalVec.append(np.round(self.state[0][0], 2))
        self.velVec.append(np.round(self.state[0][1], 2))
        self.accVec.append(np.round(self.state[0][2], 1))
        self.jerkVec.append(self.jerkActionMap[action])
        # self.actionVec.append(action)

        # Distance to go plot
        self.axDisToGo.set_xlabel(" t - s")
        self.axDisToGo.set_ylabel(" d - m")
        self.axDisToGo.plot(self.timeVec, self.distToGoalVec, 'b')
        self.axDisToGo.set_xlim([-1, Tf])  # Time limit
        self.axDisToGo.set_ylim([-self.env.target, 2*self.env.target])
        self.axDisToGo.grid(True)

        # Velocity plot
        self.axVec.set_xlabel(" t - s")
        self.axVec.set_ylabel(" v - m/s")
        self.axVec.plot(self.timeVec, self.velVec, 'b')
        self.axVec.set_xlim([-1, Tf])  # Time limit
        self.axVec.set_ylim([-2, self.env.vMax])
        self.axVec.grid(True)

        # Acceleration plot
        self.axAcc.set_xlabel(" t - s")
        self.axAcc.set_ylabel(" a - m/s^2")
        self.axAcc.plot(self.timeVec, self.accVec, 'b')
        self.axAcc.set_xlim([-1, Tf])  # Time limit
        self.axAcc.set_ylim([-(self.env.aMax + 2), self.env.aMax + 2])
        self.axAcc.grid(True)

        # self.axAct.set_xlabel(" t - s")
        # self.axAct.set_ylabel(" Action [0, 1, 2]")
        # self.axAct.plot(self.timeVec, self.actionVec, 'b')

        # Jerk input - Action plot
        self.axAct.set_xlabel(" t - s")
        self.axAct.set_ylabel(" j - m/s^2")
        self.axAct.plot(self.timeVec, self.jerkVec, 'b')
        self.axAct.set_xlim([-1, Tf])  # Time limit
        self.axAct.set_ylim([-(self.env.jMax + 2), self.env.jMax + 2])
        self.axAct.grid(True)

        if self.verbose:
            print("Time: {}, Distance to goal: {}, Velocity: {}, Acceleration: {}, Jerk Input: {}".format(
                self.timeVec[-1], self.distToGoalVec[-1], self.velVec[-1], self.accVec[-1],
                self.jerkVec[-1]))

        return []
