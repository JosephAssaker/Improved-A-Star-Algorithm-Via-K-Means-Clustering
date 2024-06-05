import xml.etree.ElementTree as ET
import time
from graphics import *
from math import sqrt
from random import randint
from random import uniform

tree = ET.parse('maps/map.net.xml')
root = tree.getroot()

winWidth = 1600
winHeight = 800
winWH_k = 600
win = GraphWin('My Graph: ', winWidth, winHeight)
win_kmeans = GraphWin('K means: ', winWH_k, winWH_k)
invAxe = winHeight/2

nodes = []
nodesID = []
cars = []
edges = []

M = []  # Correspondance Matrix
H = []  # Weights vector - Heuristic Function


def get_neighbours(node):
    neighbours = []
    for j in range(len(M)):
        if(M[node][j] != 0):
            neighbours.append(j)
    return neighbours


def get_neighbours_w_weights(node):
    neighbours = []
    for j in range(len(M)):
        if(M[node][j] != 0):
            neighbours.append([j, M[node][j]])
    return neighbours


def get_reachable_from_w_weights(node):
    neighbours = []
    for j in range(len(M)):
        if(M[j][node] != 0):
            neighbours.append([j, M[j][node]])
    return neighbours
   

def a_star_optimized(matrix, start, end):
    l = [start]
    output = []
    b = False

    while True:
        current_node = l.pop(len(l)-1)
        if not (current_node in output):
            output.append(current_node)
        else:
            continue
        if current_node == end:
            b = True
            break
        neighbours = get_neighbours_w_weights(current_node)
        for i in range(len(neighbours)):
            neighbours[i] = [neighbours[i][0],
                             neighbours[i][1], H[neighbours[i][0]]]
        for n in neighbours:
            if n[1] + n[2] == H[current_node]:
                l.append(n[0])
                break
        if not l:
            break

    if b:
        print('Reached Destination!\nPath is: ', end='')
        for node in output:
            print(node, end=' ')
        print()
    else:
        print('Destination not reachable...\n')
    return output


def a_star_optimized_with_kmeans(matrix, start, end, k_groups):
    l = [start]
    output = []
    b = False

    while True:
        current_node = l.pop(len(l)-1)
        if not (current_node in output):
            output.append(current_node)
        else:
            continue
        if current_node == end:
            b = True
            break
        neighbours = get_neighbours_w_weights(current_node)
        for i in range(len(neighbours)):
            neighbours[i] = [neighbours[i][0],
                             neighbours[i][1], H[neighbours[i][0]]] #neighbour number: x, g(x), h(x)

        groups = []
        for i in range(3):
            groups.append([])

        for i in range(len(neighbours)):
            for j in range(len(k_groups)):
                if neighbours[i][0] in k_groups[j]:
                    groups[j].append(i)

        for i in range(len(k_groups)):
            if groups[i]:
                min = 999999
                min_node = -1
                for j in range(len(groups[i])):
                    temp = neighbours[groups[i][j]][1] + neighbours[groups[i][j]][2]
                    if temp < min and not (neighbours[groups[i][j]][0] in output) and neighbours[groups[i][j]][2] < H[current_node] and neighbours[groups[i][j]][2] != -1:
                        min = temp
                        min_node = groups[i][j]
                if min == 999999:
                    continue
                l.append(neighbours[min_node][0])
                break

        if not l:
            break

    if b:
        print('Reached Destination!\nPath is: ', end='')
        for node in output:
            print(node, end=' ')
        print()
    else:
        print('Destination not reachable...\n')
    return output


def euclidian_dist(A, B):
    return sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


def mean(points):
    x = 0
    y = 0
    for i in range(len(points)):
        x += H_normalized[points[i]]
        y += cars[points[i]]
    x /= len(points)
    y /= len(points)
    return x, y


def convergence(A, B):
    if len(A) == 0 or len(B) == 0 or len(A) != len(B):
        print('A and B does not conform the basic requirements. Func: Convergence')
        exit(-1)
    b = True
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            print('A and B does not conform the basic requirements. Func: Convergence')
            exit(-1)
        for j in range(len(A[i])):
            if A[i][j] != B[i][j]:
                b = False
    return b


def k_means(k):
    centroids = []
    prev_centroids = []
    k_groups = []
    for i in range(k):
        centroids.append([uniform(0, 1), uniform(0, 1)])
        prev_centroids.append([-1,-1])

    while not convergence(prev_centroids, centroids):
        for i in range(k):
            prev_centroids[i][0] = centroids[i][0]
            prev_centroids[i][1] = centroids[i][1]
        k_groups.clear()
        for i in range(k):
            k_groups.append([])
        for i in range(len(nodesID)):
            min = euclidian_dist(
                [H_normalized[i], cars[i]], [centroids[0][0], centroids[0][1]])
            min_centroid = 0
            for j in range(1, k):
                temp = euclidian_dist(
                    [H_normalized[i], cars[i]], [centroids[j][0], centroids[j][1]])

                if temp < min:
                    min = temp
                    min_centroid = j
            k_groups[min_centroid].append(i)

        for i in range(k):
            if(len(k_groups[i]) == 0):
                continue
            x, y = mean(k_groups[i])
            centroids[i][0] = x
            centroids[i][1] = y
    return k_groups, centroids


def speed(i, j):
    traffic = (cars[i] + cars[j]) / 2
    if traffic < 0.2:
        return 16.666 #60 Km/h
    if traffic < 0.5:
        return 6.944 #25 Km/h
    if traffic < 0.8:
        return 2.777 #10 Km/h
    return 1.111 # 4 Km/h


xMax = 0
yMax = 0

# Getting the maximum y and x coordinates in the input map so that the map properly scale to the windown (stretch fill).
for node in root.iter('junction'):
    A = node.attrib
    if A.get('type') == 'internal' or A.get('type') == 'dead_end':
        continue
    x = int(float(A.get('x')))
    y = int(float(A.get('y')))
    if x > xMax:
        xMax = x
    if y > yMax:
        yMax = y

# Filling the nodes, nodesID and the cars lists.
for node in root.iter('junction'):
    A = node.attrib
    if A.get('type') == 'internal' or A.get('type') == 'dead_end':
        continue
    x = int(float(A.get('x'))) / xMax * winWidth
    y = int(float(A.get('y'))) / yMax * winHeight
    y_inv = 2*invAxe - y
    nodes.append(Circle(Point(x, y_inv), 2))
    cars.append(randint(0, 120))
    nodesID.append(A.get('id'))

# Initializing list H to -1s and the matrix M to 0s.
for i in range(len(nodesID)):
    M.append([])
    H.append(-1)
    for j in range(len(nodesID)):
        M[i].append(0)

# Populating matrix M such that M[i][j] = distance between i and j, and the list edges.
for child in root.iter('edge'):
    A = child.attrib
    if len(A) > 2:
        F = A.get('from')
        T = A.get('to')
        if not (F in nodesID) or not (T in nodesID):
            continue 
        Findex = nodesID.index(F)
        Tindex = nodesID.index(T)
        edges.append(Line(nodes[Findex].getCenter(),
                          nodes[Tindex].getCenter()))
        weight = 0
        for lane in child.iter('lane'):
            a = lane.attrib
            weight = float(a.get('length'))
            break
        M[Findex][Tindex] = weight

# Drawing the map's edges.
for edge in edges:
    edge.setWidth = 1
    edge.draw(win)

start = 2228 #randint(0, len(nodesID) - 1)
end = 680 #randint(0, len(nodesID) - 1)

start_time = time.time()

# Populating weights from destination using BFS
l = [[end, 0]]
H[end] = 0
while True:
    current_node = l.pop(len(l) - 1)
    neighbours = get_reachable_from_w_weights(current_node[0])
    for n in neighbours:
        node_weight = H[n[0]]
        if (node_weight == -1) or (node_weight > n[1] + current_node[1]):
            H[n[0]] = n[1] + current_node[1]
            l.insert(0, [n[0], n[1] + current_node[1]])
    if not l:
        break
print('Weights distribution using BFS took {} seconds'.format(time.time() - start_time))

start_time = time.time()
path = a_star_optimized(M, start, end)
print('A* optimized done in: {} seconds.'.format(time.time() - start_time))


H_normalized = []

# Duplicating the W list in, the next normalized, W_normalized as the initial list W will still be needed in the A* algorithm. 
for i in range(len(H)):
    H_normalized.append(H[i])

xMax_k = 0
yMax_k = 0
for i in range(len(nodesID)):
    if H_normalized[i] > xMax_k:
        xMax_k = H_normalized[i]
    if cars[i] > yMax_k:
        yMax_k = cars[i]

# Normalizing the two inputs of the Kmeans algorithm such that for any i in [0, len(nodes) - 1] 0 <= W_normalized[i] <= 1 and 0 <= cars[i] <= 1.
for i in range(len(nodesID)):
    H_normalized[i] /= xMax_k
    cars[i] /= yMax_k

# Choosing to have an output of three clusters from the Kmeans algorithm.
K = 3

k_groups, centroids = k_means(K)

min = euclidian_dist(centroids[0], [0, 0])
max = min
min_cent = 0
max_cent = 0
for i in range(1, K):
    temp = euclidian_dist(centroids[i], [0, 0])
    if temp < min:
        min = temp
        min_cent = i
    if temp > max:
        max = temp
        max_cent = i

# Reorganizing the clusters and their respective centroids in ascending order.
temp_cent = []
temp_cent.append(centroids[min_cent])
temp_cent.append(centroids[3 - max_cent - min_cent])
temp_cent.append(centroids[max_cent])
centroids = temp_cent

temp_grp = []
temp_grp.append(k_groups[min_cent])
temp_grp.append(k_groups[3 - max_cent - min_cent])
temp_grp.append(k_groups[max_cent])
k_groups = temp_grp

print()
for i in range(K):
    temp = euclidian_dist(centroids[i], [0, 0])
    print('Euclidian dist from [0, 0] of centroid {} of [{},{}] is: {}'.format(i,centroids[i][0],centroids[i][1],temp))

# Displaying the centroids on the Kmeans window.
print()
for i in range(K):
    print('Group {} has {} members\n'.format(i, len(k_groups[i])))

    centroids[i][0] = centroids[i][0] * winWH_k
    centroids[i][1] = centroids[i][1] * winWH_k
    centroids[i][1] = winWH_k - centroids[i][1]

    C = Circle(Point(centroids[i][0], centroids[i][1]), 10)
    C.setFill('green')
    C.draw(win_kmeans)

Y_axis = []

for i in range(len(nodesID)):
    H_normalized[i] *= winWH_k
    Y_axis.append(cars[i] * winWH_k)
    Y_axis[i] = winWH_k - Y_axis[i]

# Displaying the nodes as well as their belonging (represented by their color) on the Kmeans window.
for i in range(len(nodesID)):
    C = Circle(Point(H_normalized[i], Y_axis[i]), 2)
    if(i in k_groups[0]):
        C.setFill('blue')
    elif(i in k_groups[1]):
        C.setFill('yellow')
    elif(i in k_groups[2]):
        C.setFill('red')
    else:
        C.setFill('black')
    C.draw(win_kmeans)

optimal_path = a_star_optimized_with_kmeans(M, start, end, k_groups)

length = 0
for i in range(len(path)-1):
    length += M[path[i]][path[i+1]]
print('A* path length is: {} meters.'.format(length))

time_needed = 0
for i in range(len(path)-1):
    time_needed += M[path[i]][path[i+1]] / speed(path[i], path[i+1])
print('A* path would take {} minutes {} seconds'.format(time_needed // 60, time_needed - 60 * (time_needed//60)))

length = 0
for i in range(len(optimal_path)-1):
    length += M[optimal_path[i]][optimal_path[i+1]]
print('A* with Kmeans path length is: {} meters.'.format(length))

time_needed = 0
for i in range(len(optimal_path)-1):
    time_needed += M[optimal_path[i]][optimal_path[i+1]] / speed(optimal_path[i], optimal_path[i+1])
print('A* with Kmeans path would take {} minutes {} seconds'.format(time_needed // 60, time_needed - 60 * (time_needed//60)))

# Displaying nodes on the Graph window, as well as their belonging in case they belong to a certain path.
for i in range(len(nodes)):
    if i == path[0] or i == path[len(path)-1]:
        nodes[i].setFill('green')
        nodes[i].setOutline('green')
        nodes[i].setWidth(15)
    elif i in optimal_path:
        nodes[i].setFill('red')
    elif i in path:
        nodes[i].setFill('yellow')
    else:
        nodes[i].setFill('blue')
    nodes[i].draw(win)

# Waiting for user input on both windows to terminate the program.
win.getMouse()
win.close()
win_kmeans.getMouse()
win_kmeans.close()
