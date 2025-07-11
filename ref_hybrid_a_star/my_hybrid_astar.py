"""
Hybrid A*
@author: Huiming Zhou
"""

import os
import sys
import math
import heapq
from heapdict import heapdict
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

import draw as draw
import reeds_shepp as rs

class C:  # Parameter config
    PI = math.pi

    XY_RESO = 0.2  # [m]
    YAW_RESO = np.deg2rad(15.0)  # [rad]
    MOVE_STEP = 0.2  # [m] path interporate resolution
    N_STEER = 10.0  # steer command number
    MAX_ANGULAR_VELOCITY = 0.5   # [rad/s] maximum angular velocity
    MIN_ANGULAR_VELOCITY = -0.5  # [rad/s] minimum angular velocity
    MAX_CURVATURE_RADIUS = 0.5  # [m] maximum curvature radius
    COLLISION_CHECK_STEP = 2    # skip number for collision check

    GEAR_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 50.0  # backward penalty cost
    ANGULAR_VELOCITY_CHANGE_COST = 2.0  # angular velocity change penalty cost
    H_COST = 10.0  # Heuristic cost penalty cost

    RADIUS = 0.4 # [m] radius of vehicle
    RF = 0.6  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.2  # [m] distance from rear to vehicle back end of vehicle
    W = 0.6  # [m] width of vehicle
    WD = 0.6  # [m] distance between left-right wheels
    WB = 0.6  # [m] Wheel base
    TR = 0.2  # [m] Tyre radius
    TW = 0.2  # [m] Tyre width

class HolonomicNode:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node

class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = heapdict()

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        self.queue[item] = priority  # push 

    def get(self):
        return self.queue.popitem()[0]  # pop out element with smallest priority

def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion

def obstacles_map(P, rr):
    # Compute Grid Index for obstacles
    ox_grid = [round(x / P.xyreso) for x in P.ox]
    oy_grid = [round(y / P.xyreso) for y in P.oy]

    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox_grid, oy_grid):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.xyreso:
                    obsmap[x][y] = True
                    break
    return obsmap

def calc_holonomic_heuristic_with_obstacle(node, P, radius):
    
    n_goal = HolonomicNode(round(node.x[-1] / P.xyreso), round(node.y[-1] / P.xyreso), 0.0, -1)

    motion = get_motion()

    obsmap = obstacles_map(P, radius)

    open_set, closed_set = dict(), dict()
    open_set[calc_holonomic_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_holonomic_index(n_goal, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(motion)):
            node = HolonomicNode(n_curr.x + motion[i][0],
                        n_curr.y + motion[i][1],
                        n_curr.cost + u_cost(motion[i]), ind)

            if not check_holonomic_node(node, P, obsmap):
                continue

            n_ind = calc_holonomic_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_holonomic_index(node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap

def check_holonomic_node(node, P, obsmap):
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        return False

    return True

def u_cost(u):
    return math.hypot(u[0], u[1])

def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox_grid, oy_grid, xyreso, yawreso, radius):
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

    P = calc_parameters(ox_grid, oy_grid, xyreso, yawreso)

    hmap = calc_holonomic_heuristic_with_obstacle(ngoal, P, radius)

    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    count = 1
    while True:
        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P)

        if update:
            print(f"found Reeds-Shepp path in {count} iterations")
            fnode = fpath
            break

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not node:
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
        count += 1
    return extract_path(closed_set, fnode, nstart)


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, direc, cost)

    return path


def calc_next_node(n_curr, c_id, u, d, P):
    step = C.XY_RESO * 2

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP * u)]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP * u))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None

    cost = 0.0

    if d > 0:
        direction = 1
        cost += abs(step)
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST

    if direction != n_curr.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.ANGULAR_VELOCITY_CHANGE_COST * abs(n_curr.steer - u)  # velocity change penalty
    cost = n_curr.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node


def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False

    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)

    nodex = [xlist[k] for k in ind]
    nodey = [ylist[k] for k in ind]
    nodeyaw = [yawlist[k] for k in ind]

    if is_collision(nodex, nodey, nodeyaw, P):
        return False

    return True


def update_node_with_analystic_expantion(n_curr, ngoal, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path)
    fpind = calc_index(n_curr, P)
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    #  Find all possible reeds-shepp paths between current and goal node
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, C.MAX_CURVATURE_RADIUS, step_size=C.MOVE_STEP)

    if not paths:
        return None

    # Find path with lowest cost considering non-holonomic constraints
    pq = QueuePrior()
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))

    # Find first path in priority queue that is collision free
    while not pq.empty():
        path = pq.get()
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]

        if not is_collision(pathx, pathy, pathyaw, P):
            return path

    return None


def is_collision(x, y, yaw, P):
    for ix, iy, iyaw in zip(x, y, yaw):
        car_length = C.RF + C.RB
        safety_margin = P.xyreso
        r = max(car_length / 2.0, C.W / 2.0) + safety_margin
        dl = (C.RF - C.RB) / 2.0

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        for i in ids:
            xo = P.ox[i] - cx
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r  and abs(dy) < C.W / 2 + safety_margin:
                return True

    return False

def calc_rs_path_cost(rspath):
    cost = 0.0

    # Distance cost
    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    # Direction change cost
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    return cost


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_motion_set():
    angular_velocities = np.arange(C.MIN_ANGULAR_VELOCITY, C.MAX_ANGULAR_VELOCITY, (C.MAX_ANGULAR_VELOCITY - C.MIN_ANGULAR_VELOCITY) / C.N_STEER)
    
    steer = list(angular_velocities) + [0.0] + list(-angular_velocities)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True

def calc_holonomic_index(node, P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)

def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


def calc_parameters(ox_grid, oy_grid, xyreso, yawreso):

    minx = round(min(ox_grid))
    miny = round(min(oy_grid))
    maxx = round(max(ox_grid))
    maxy = round(max(oy_grid))
    
    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    ox = [i * xyreso for i in ox_grid]
    oy = [i * xyreso for i in oy_grid]

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])

    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)


def draw_car(x, y, yaw, steer, color='black'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)


def generate_obstacle_in_grid_map():
    # Build Map
    obstacleX, obstacleY = [], []

    rectangle = [-2, 5, -2, 5] # x_min, x_max, y_min, y_max
    rectangle_grid_index = [round(i/C.XY_RESO) for i in rectangle] # x_min, x_max, y_min, y_max in grid

    for i in range(rectangle_grid_index[0],rectangle_grid_index[1]+1):
        obstacleX.append(i)
        obstacleY.append(rectangle_grid_index[2])

        obstacleX.append(i)
        obstacleY.append(rectangle_grid_index[3])

    for i in range(rectangle_grid_index[2]+1,rectangle_grid_index[3]):
        obstacleX.append(rectangle_grid_index[0])
        obstacleY.append(i)

        obstacleX.append(rectangle_grid_index[1])
        obstacleY.append(i)
    
    center_of_map = [round((rectangle_grid_index[0] + rectangle_grid_index[1])/2), round((rectangle_grid_index[2] + rectangle_grid_index[3])/2)]

    for i in range(rectangle_grid_index[2]+1, round(center_of_map[1]+1)):
        obstacleX.append(center_of_map[0]+2)
        obstacleY.append(i)

    for i in range(round(center_of_map[0]-3),rectangle_grid_index[1]):
        obstacleX.append(center_of_map[0]-4)
        obstacleY.append(i)


    return obstacleX, obstacleY


def main():
    print("start!")
    sx, sy, syaw0 = -0.5, 2, np.deg2rad(90.0)
    gx, gy, gyaw0 = 3.0, 0, np.deg2rad(180.0)

    ox_grid, oy_grid = generate_obstacle_in_grid_map()
    ox, oy = [i * C.XY_RESO for i in ox_grid], [i * C.XY_RESO for i in oy_grid]

    t0 = time.time()
    path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
                                 ox_grid, oy_grid, C.XY_RESO, C.YAW_RESO, C.RADIUS)
    t1 = time.time()
    print("running time: ", t1 - t0)

    if not path:
        print("Searching failed!")
        return

    print(path)
    x = path.x
    y = path.y
    yaw = path.yaw
    direction = path.direction

    print(len(x))
    for k in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, "sk")
        plt.plot(x, y, linewidth=1.5, color='r')

        if k < len(x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction[k]))
        else:
            steer = 0.0

        draw_car(gx, gy, gyaw0, 0.0, 'dimgray')
        draw_car(x[k], y[k], yaw[k], steer)
        plt.title("Hybrid A*")
        plt.axis("equal")
        plt.pause(0.0001)

    plt.show()
    print("Done!")


if __name__ == '__main__':
    main()
