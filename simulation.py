from math import pi
from env import Robot


def init():
    ax.add_patch(agent.patch)
    pathLine.set_data([], [])
    return []

def animate_robot(robot_patch, x_, y_, theta_):
    robot_patch.set_xy([x_ - Robot.xSize / 2, y_ - Robot.ySize / 2])
    robot_patch.set_angle(theta_ * 180 / pi)


def animate_all(_, dt=0.05):
    global time_t, xVec, xAcc, tVec, case, replanFlag
    time_t += dt
    # print((np.round(agent.pos()[0:2], 1) ==  np.round(agent.graph.nodes[13].nodePos(), 1) ))

    agent.move(dt)

    x_loc, y_loc, theta_loc = agent.getAgentPos(returnInXYThetaForm=True)
    animate_robot(agent.patch, x_loc, y_loc, theta_loc)

    # x_vec, x_acc = agent.getAgentVelAccInfoNormalized()
    x_vec, x_acc = agent.getAgentVelAccInfo()

    agentPos = agent.pos()


    # agentStartPos = agent.locStartPos()
    # # d_ = dist((agentPos[0], agentPos[1]), agentStartPos)

    xVec.append(x_vec)
    xAcc.append(x_acc)
    disVec.append(agentPos[0])
    tVec.append(time_t)

    axVec.clear()
    axAcc.clear()

    # axVec.set_xlim([-1, 10]) # Distance limit
    axVec.set_xlim([-1, 20]) # Time limit
    axVec.set_ylim([-2, 2])
    # axAcc.set_xlim([-1, 10]) # Distance limit
    axAcc.set_xlim([-1, 20]) # Time limit
    axAcc.set_ylim([-2, 2])

    axVec.set_ylabel(" v(t) - m/s")
    axAcc.set_ylabel(" a(t) - m/s^2")

    if plotAgainstDistance:
        axAcc.set_xlabel(" d(t) - m")
        axVec.plot(disVec, xVec, 'b')
        axAcc.plot(disVec, xAcc, 'r')
    else:
        axAcc.set_xlabel(" t - s")
        axVec.plot(tVec, xVec, 'b')
        axAcc.plot(tVec, xAcc, 'r')
        # axAcc.set_xlabel(" v(t) - m/s")
        # axAcc.plot(xVec, xAcc, 'r')

    axVec.grid(which='both')
    axVec.minorticks_on()
    axAcc.grid(which='both')
    axAcc.minorticks_on()

    locGoalPos = agent.locGoalPos()
    pathLine.set_data([x_loc, locGoalPos[0]], [y_loc, locGoalPos[1]])

    agentVecAccInfo = agent.getAgentVelAccInfo()
    time_text.set_text("Time = %.1f sec" % time_t)
    vel_text.set_text("x vel = %.1f m/s" % agentVecAccInfo[0])
    acc_text.set_text("x acc = %.1f m/s^2" % agentVecAccInfo[1])

    return []

def start_sim(ag, cs):
    global agent, fig, ax, axVec, axAcc, time_t, time_text, vel_text, acc_text, xVec, xAcc, tVec, pathLine, disVec, replanFlag
    global plotAgainstDistance, case, goalChangeNode

    case = cs
    plotAgainstDistance = False

    goalChangeNode = 12
    agent = ag

    xVec = []
    xAcc = []
    tVec = []
    disVec = []

    x_vec, x_acc = agent.getAgentVelAccInfo()
    time_t = 0
    xVec.append(x_vec)
    xAcc.append(x_acc)
    tVec.append(time_t)
    disVec.append(0)

    fig, ax, axVec, axAcc, time_text, vel_text, acc_text = Robot.graph.plot_graph()

    # fig.suptitle(fig_title, fontsize=20)
    replanFlag = True
    pathLine, = ax.plot([], [], lw=2)

    return fig
