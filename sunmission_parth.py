import cv2.aruco as aruco
import numpy as np
import math
import math
import cv2 as cv
import gym
import pixelate_arena
import time
import pybullet as p
import pybullet_data
import os
from collections import deque, namedtuple
import vectormath as vmath
from collections import deque, namedtuple
import gym



kernel=np.ones((3,3),np.uint8)
size = 600
inf = 999999999

blocks = []
for i in range(13):
  for j in range(25):
    blocks.append(
    {
        "block_num" : (i*25+j),
        "color":'black',
        'weight':inf ,
        "x-coordinate" : j,
        "y-coordinate" : i
    }
    )


env = gym.make("pixelate_arena-v0")
#env.remove_car()
img = env.camera_feed()
#env.respawn_car()

#img = cv.imread('qwer.jpeg')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#masks:
red_mask = cv.inRange(img, np.array([0, 0, 150]), np.array([25, 25, 255]))
yellow_mask = cv.inRange(img, np.array([0, 100, 100]), np.array([20, 255, 255]))
green_mask = cv.inRange(img_hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
purple_mask = cv.inRange(img_hsv, np.array([120, 0, 0]), np.array([150, 255, 255]))
pink_mask = cv.inRange(img_hsv, np.array([150, 0, 0]), np.array([170, 255, 255]))
white_mask = cv.inRange(img_hsv, np.array([0, 0, 200]), np.array([0, 100, 255]))
red_mask =  cv.dilate(red_mask,kernel,iterations=1)
blue_mask = cv.inRange(img, np.array([150, 0, 0]), np.array([255, 70, 25]))

cv.imshow('img', img)
#cv.imshow('red_mask', red_mask)
#cv.imshow('yellow_mask', yellow_mask)
#cv.imshow('green_mask', green_mask)
#cv.imshow('pink_mask', pink_mask)
#cv.imshow('purple_mask', purple_mask)
#cv.imshow('white_mask', white_mask)
#cv.imshow('blue_mask', blue_mask)


def rescale_coordinatesX(cX):
    if cX<=44:
        return 0

    if cX>44 and cX<=66:
        return 1

    if cX>65 and cX<=88:
        return 2

    if cX>88 and cX<=110:
        return 3

    if cX>110 and cX<=132:
       return 4

    if cX>132 and cX<=154:
       return 5

    if cX>154 and cX<=176:
       return 6

    if cX>176 and cX<=198:
       return 7

    if cX>198 and cX<=220:
       return 8

    if cX>220 and cX<=242:
       return 9

    if cX>242 and cX<=264:
       return 10

    if cX>264 and cX<= 286:
       return 11

    if cX>286 and cX<=308:
       return 12

    if cX>308 and cX<=330:
       return 13

    if cX>330 and cX<=352:
       return 14

    if cX>352 and cX<=374:
       return 15

    if cX>374 and cX<=396:
       return 16

    if cX>396 and cX<=418:
       return 17

    if cX>418 and cX<=440:
       return 18

    if cX>440 and cX<=462:
       return 19

    if cX>462 and cX<=484:
       return 20

    if cX>484 and cX<=506:
       return 21

    if cX>506 and cX<=528:
       return 22

    if cX>528 and cX<=550:
       return 23

    if cX>550:
       return 24


def rescale_coordinatesY(cY):
    if cY<=85.25:
        return 0

    if cY>85.25 and cY<=123.75:
        return 1

    if cY>123.75 and cY<=162.25:
        return 2

    if cY>162.25 and cY<=200.75:
        return 3

    if cY>200.75 and cY<=239.25:
        return 4

    if cY>239.25 and cY<=277.75:
        return 5

    if cY>277.75 and cY<=316.25:
        return 6

    if cY>316.25 and cY<=354.75:
        return 7

    if cY>354.75 and cY<=393.25:
        return 8

    if cY>393.25 and cY<=431.75:
        return 9

    if cY>431.75 and cY<=470.25:
        return 10

    if cY>470.25 and cY<=508.75:
        return 11

    if cY>508.75 :
        return 12




def find_coordinates(colored_blocks):
    contours, _ = cv.findContours(colored_blocks, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    list_of_coordinates = []
    for cnt in contours:
        M = cv.moments(cnt)
        # calculate x,y coordinate of center
        if(M["m00"]!=0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            exactcX = cX
            exactcY = cY
            cX = rescale_coordinatesX(cX)
            cY = rescale_coordinatesY(cY)
        list_of_coordinates.append((cX, cY, exactcX, exactcY))
        
    return list_of_coordinates


xy_of_green = find_coordinates(green_mask)
xy_of_red = find_coordinates(red_mask)
xy_of_yellow = find_coordinates(yellow_mask)
xy_of_purple = find_coordinates(purple_mask)
xy_of_pink = find_coordinates(pink_mask)
xy_of_white = find_coordinates(white_mask)
xy_of_blue = find_coordinates(blue_mask)

for xy in xy_of_green:
    blocknumber = xy[1]*25 + xy[0]

    blocks[blocknumber].update( 
    {
        "block_num" : blocknumber,
        "color":'green',
        'weight':4 ,
        "x-coordinate" : xy[0],
        "y-coordinate" : xy[1],
        "exact_x-coordinate" : xy[2],
        "exact_y-coordinate" : xy[3]
    }
    )
    #print(blocks[blocknumber])

path1_startnode = 162

for xy in xy_of_purple:
    blocknumber = xy[1]*25 + xy[0]

    blocks[blocknumber].update( 
    {
        "block_num" : blocknumber,
        "color":'purple',
        'weight':3 ,
        "x-coordinate" : xy[0],
        "y-coordinate" : xy[1],
        "exact_x-coordinate" : xy[2],
        "exact_y-coordinate" : xy[3]
    }
    )
    #print(blocks[blocknumber])

for xy in xy_of_yellow:
    blocknumber = xy[1]*25 + xy[0]

    blocks[blocknumber].update( 
    {
        "block_num" : blocknumber,
        "color":'yellow',
        'weight':2 ,
        "x-coordinate" : xy[0],
        "y-coordinate" : xy[1],
        "exact_x-coordinate" : xy[2],
        "exact_y-coordinate" : xy[3]
    }
    )
    #print(blocks[blocknumber])

for xy in xy_of_pink:
    blocknumber = xy[1]*25 + xy[0]

    path2_endnode = blocknumber
    path3_startnode = blocknumber

    blocks[blocknumber].update( 
    {
        "block_num" : blocknumber,
        "color":'pink',
        'weight':1 ,
        "x-coordinate" : xy[0],
        "y-coordinate" : xy[1],
        "exact_x-coordinate" : xy[2],
        "exact_y-coordinate" : xy[3]
    }
    )
    #print(blocks[blocknumber])

for xy in xy_of_white:
    blocknumber = xy[1]*25 + xy[0]

    blocks[blocknumber].update( 
    {
        "block_num" : blocknumber,
        "color":'white',
        'weight':1 ,
        "x-coordinate" : xy[0],
        "y-coordinate" : xy[1],
        "exact_x-coordinate" : xy[2],
        "exact_y-coordinate" : xy[3]
    }
    )
    #print(blocks[blocknumber])

for xy in xy_of_red:
    blocknumber = xy[1]*25 + xy[0]

    if blocknumber!=162:
        path1_endnode = blocknumber
        path2_startnode = blocknumber

    blocks[blocknumber].update( 
    {
        "block_num" : blocknumber,
        "color":'red',
        'weight':0 ,
        "x-coordinate" : xy[0],
        "y-coordinate" : xy[1],
        "exact_x-coordinate" : xy[2],
        "exact_y-coordinate" : xy[3]
    }
    )
    print(blocks[blocknumber])


for xy in xy_of_blue:
    blocknumber = xy[1]*25 + xy[0]
    path3_endnode = blocknumber

    print(path3_endnode)



graphing = []

for tile in blocks:
    if tile["color"]!="black":

        if tile["x-coordinate"]>=2 : 
            graphing.append([tile["block_num"],   blocks[tile["block_num"]-2]["block_num"], blocks[tile["block_num"]-2]["weight"], "left"])

        if tile["x-coordinate"]<=22 : 
            graphing.append([tile["block_num"],   blocks[tile["block_num"]+2]["block_num"], blocks[tile["block_num"]+2]["weight"], "right"])

        if tile["y-coordinate"]>=1 : 
            if tile["x-coordinate"]>=1:
                graphing.append([tile["block_num"],   blocks[tile["block_num"]-26]["block_num"], blocks[tile["block_num"]-26]["weight"], "top_left"])
            if tile["x-coordinate"]<=23:
                graphing.append([tile["block_num"],   blocks[tile["block_num"]-24]["block_num"], blocks[tile["block_num"]-24]["weight"], "top_right"])

        if tile["y-coordinate"]<=11 : 
            if tile["x-coordinate"]<=23:
                graphing.append([tile["block_num"],   blocks[tile["block_num"]+26]["block_num"], blocks[tile["block_num"]+26]["weight"], "bottom_right"])
            if tile["x-coordinate"]>=1:
                graphing.append([tile["block_num"],   blocks[tile["block_num"]+24]["block_num"], blocks[tile["block_num"]+24]["weight"], "bottom_left"])


    
#for x in graphing:
#    print(x)
    
#print(graphing)    
#print(blocks)

for x in blocks:
    print(x)   

#print(path1_startnode)
#print(path1_endnode)
#print(path2_startnode)
#print(path2_endnode)
#print(path3_startnode)
#print(path3_endnode)



Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
  return Edge(start, end, cost)


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            if distances[current_vertex] == inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path


weights = graphing

weights_new = []

for i in weights:
    element = (str(i[0]),str(i[1]),i[2])
    weights_new.append(element)

graph = Graph(weights_new)

# print(weights_new)

path1 = list(graph.dijkstra(str(path1_startnode),str(path1_endnode)))
path2 = list(graph.dijkstra(str(path2_startnode),str(path2_endnode)))
path3 = list(graph.dijkstra(str(path3_startnode),str(path3_endnode)))
print(path1)
print(path2)
print(path3)


#############################################################################################################################

parent_path = os.path.dirname(os.getcwd())
os.chdir(parent_path)
time.sleep(0.5)

ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
rvecs, tvecs = None, None


img = env.camera_feed()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
print('ID: {}; Corners: {}'.format(ids, corners))

print(corners[0][0][0])


#y =0
#while True:
#    p.stepSimulation()
#    env.move_husky(20, 20, 20, 20)
#    y = y +1
#    #print(y)
#    if y>=428:
#        break


for target_tile in path1[1:]:

    print( blocks[int(target_tile) ])

    print(blocks[int(target_tile) ]["exact_x-coordinate"])
    print(blocks[int(target_tile) ]["exact_y-coordinate"])


    while True:
        while True:
            img = env.camera_feed()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if(corners[0][0].all()):
                center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
                y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]
                break 
            else:
                for x in range(2):
                    p.stepSimulation()
                    env.move_husky(1,-1,1,-1)

        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X

        vector_1 = [x_vector_bot, y_vector_bot]
        vector_2 = [x_vector_rel, y_vector_rel]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_2, unit_vector_1)
        angle_rel = (np.arccos(dot_product))*(180/np.pi) 

        print(angle_rel)
        print("#################################")
        print(vector_1)
        print(vector_2)
        print(angle_rel)
        print("#################################")



        if angle_rel>=0:
            if abs(angle_rel) <= 20:
                x =0
                while True:
                    x = x+1
                    if x%5==0:
                        img = env.camera_feed()
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                        if(corners[0][0].all()):
                            center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                            center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
    
                            y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                            x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                        else:
                            for x in range(2):
                                p.stepSimulation()
                                env.move_husky(1,-1,1,-1)
                                img = env.camera_feed()
                                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                                agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                    
                        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
                        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X
                        vector_1 = [x_vector_bot, y_vector_bot]
                
                        vector_2 = [x_vector_rel, y_vector_rel]
                        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                        dot_product = np.dot(unit_vector_2, unit_vector_1)
                        angle_rel = (np.arccos(dot_product))*(180/np.pi) 

                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)

                    if abs(angle_rel)<=2:
                        for rando in range(25):
                            p.stepSimulation()
                            env.move_husky(0,0,0,0)
                        print("===================")
                        print(angle_rel)
                        print("===================")
                        break
                break

            elif abs(angle_rel) <= 40:
                for x in range(50):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            elif abs(angle_rel) <= 90:
                for x in range(80):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            else:
                for x in range(180):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(40):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

        if angle_rel<0:
            if abs(angle_rel) <= 15:
                while True:
                    img = env.camera_feed()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                    if(corners[0][0].all()):
                        center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                        center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4

                        y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                        x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                    else:
                        for x in range(2):
                            p.stepSimulation()
                            env.move_husky(1,-1,1,-1)
                            img = env.camera_feed()
                            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                    
                    y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
                    x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X
                    vector_1 = [x_vector_bot, y_vector_bot]
                
                    vector_2 = [x_vector_rel, y_vector_rel]
                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_2, unit_vector_1)
                    angle_rel = (np.arccos(dot_product))*(180/np.pi) 

                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)

                    if abs(angle_rel)<=2:
                        for rando in range(25):
                            p.stepSimulation()
                            env.move_husky(0,0,0,0)
                        print("===================")
                        print(angle_rel)
                        print("===================")
                        break
                break

            elif abs(angle_rel) <= 40:
                for x in range(50):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)


            elif abs(angle_rel) <= 90:
                for x in range(80):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            else:
                for x in range(180):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(40):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

    

    while True:
        while True:
            img = env.camera_feed()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if(corners[0][0].all()):
                center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
                y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]
                break 
            else:
                for x in range(2):
                    p.stepSimulation()
                    env.move_husky(1,-1,1,-1)

        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X


        count =0
        while True:
            count  = count+1
            if(count%50==0):
                print(count)
            if(count%5==0):


                img = env.camera_feed()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                if(corners[0][0].all()):
                    center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                    center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4

                    y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                    x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                else:
                    for x in range(2):
                        p.stepSimulation()
                        env.move_husky(1,-1,1,-1)
                        img = env.camera_feed()
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)


            p.stepSimulation()
            env.move_husky(10, 10, 10, 10)
            y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
            x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X

            if abs(y_vector_rel)<=10 and abs(x_vector_rel)<=10:
                for rando in range(50):
                    p.stepSimulation()
                    env.move_husky(0,0,0,0)
                print("-----------------------------am I close enough??-----------------")
                break
        break



print("xxxxxxxxxxxxxxxxREACHED END OF PATH 1xxxxxxxx")
x=0

while True:
    p.stepSimulation()
    if x==10000:
        env.unlock_antidotes()
        break
    x+=1
time.sleep(1)


for target_tile in path2[1:]:

    print( blocks[int(target_tile) ])

    print(blocks[int(target_tile) ]["exact_x-coordinate"])
    print(blocks[int(target_tile) ]["exact_y-coordinate"])


    while True:
        while True:
            img = env.camera_feed()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if(corners[0][0].all()):
                center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
                y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]
                break 
            else:
                for x in range(2):
                    p.stepSimulation()
                    env.move_husky(1,-1,1,-1)

        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X

        vector_1 = [x_vector_bot, y_vector_bot]
        vector_2 = [x_vector_rel, y_vector_rel]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_2, unit_vector_1)
        angle_rel = (np.arccos(dot_product))*(180/np.pi) 

        print(angle_rel)
        print("#################################")
        print(vector_1)
        print(vector_2)
        print(angle_rel)
        print("#################################")



        if angle_rel>=0:
            if abs(angle_rel) <= 20:
                x =0
                while True:
                    x = x+1
                    if x%5==0:
                        img = env.camera_feed()
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                        if(corners[0][0].all()):
                            center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                            center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
    
                            y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                            x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                        else:
                            for x in range(2):
                                p.stepSimulation()
                                env.move_husky(1,-1,1,-1)
                                img = env.camera_feed()
                                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                                agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                    
                        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
                        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X
                        vector_1 = [x_vector_bot, y_vector_bot]
                
                        vector_2 = [x_vector_rel, y_vector_rel]
                        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                        dot_product = np.dot(unit_vector_2, unit_vector_1)
                        angle_rel = (np.arccos(dot_product))*(180/np.pi) 

                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)

                    if abs(angle_rel)<=2:
                        for rando in range(25):
                            p.stepSimulation()
                            env.move_husky(0,0,0,0)
                        print("===================")
                        print(angle_rel)
                        print("===================")
                        break
                break

            elif abs(angle_rel) <= 40:
                for x in range(50):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            elif abs(angle_rel) <= 90:
                for x in range(80):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            else:
                for x in range(180):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(40):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

        if angle_rel<0:
            if abs(angle_rel) <= 15:
                while True:
                    img = env.camera_feed()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                    if(corners[0][0].all()):
                        center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                        center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4

                        y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                        x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                    else:
                        for x in range(2):
                            p.stepSimulation()
                            env.move_husky(1,-1,1,-1)
                            img = env.camera_feed()
                            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                    
                    y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
                    x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X
                    vector_1 = [x_vector_bot, y_vector_bot]
                
                    vector_2 = [x_vector_rel, y_vector_rel]
                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_2, unit_vector_1)
                    angle_rel = (np.arccos(dot_product))*(180/np.pi) 

                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)

                    if abs(angle_rel)<=2:
                        for rando in range(25):
                            p.stepSimulation()
                            env.move_husky(0,0,0,0)
                        print("===================")
                        print(angle_rel)
                        print("===================")
                        break
                break

            elif abs(angle_rel) <= 40:
                for x in range(50):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)


            elif abs(angle_rel) <= 90:
                for x in range(80):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            else:
                for x in range(180):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(40):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

    

    while True:
        while True:
            img = env.camera_feed()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if(corners[0][0].all()):
                center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
                y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]
                break 
            else:
                for x in range(2):
                    p.stepSimulation()
                    env.move_husky(1,-1,1,-1)

        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X


        count =0
        while True:
            count  = count+1
            if(count%50==0):
                print(count)
            if(count>=250):
                break
            if(count%5==0):


                img = env.camera_feed()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                if(corners[0][0].all()):
                    center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                    center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4

                    y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                    x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                else:
                    for x in range(2):
                        p.stepSimulation()
                        env.move_husky(1,-1,1,-1)
                        img = env.camera_feed()
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)


            p.stepSimulation()
            env.move_husky(10, 10, 10, 10)
            y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
            x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X

            if abs(y_vector_rel)<=10 and abs(x_vector_rel)<=10:
                for rando in range(50):
                    p.stepSimulation()
                    env.move_husky(0,0,0,0)
                print("-----------------------------am I close enough??-----------------")
                break
        break


for target_tile in path3[1:]:

    print( blocks[int(target_tile) ])

    print(blocks[int(target_tile) ]["exact_x-coordinate"])
    print(blocks[int(target_tile) ]["exact_y-coordinate"])


    while True:
        while True:
            img = env.camera_feed()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if(corners[0][0].all()):
                center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
                y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]
                break 
            else:
                for x in range(2):
                    p.stepSimulation()
                    env.move_husky(1,-1,1,-1)

        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X

        vector_1 = [x_vector_bot, y_vector_bot]
        vector_2 = [x_vector_rel, y_vector_rel]

        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_2, unit_vector_1)
        angle_rel = (np.arccos(dot_product))*(180/np.pi) 

        print(angle_rel)
        print("#################################")
        print(vector_1)
        print(vector_2)
        print(angle_rel)
        print("#################################")



        if angle_rel>=0:
            if abs(angle_rel) <= 20:
                x =0
                while True:
                    x = x+1
                    if x%5==0:
                        img = env.camera_feed()
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                        if(corners[0][0].all()):
                            center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                            center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
    
                            y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                            x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                        else:
                            for x in range(2):
                                p.stepSimulation()
                                env.move_husky(1,-1,1,-1)
                                img = env.camera_feed()
                                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                                agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                    
                        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
                        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X
                        vector_1 = [x_vector_bot, y_vector_bot]
                
                        vector_2 = [x_vector_rel, y_vector_rel]
                        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                        dot_product = np.dot(unit_vector_2, unit_vector_1)
                        angle_rel = (np.arccos(dot_product))*(180/np.pi) 

                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)

                    if abs(angle_rel)<=2:
                        for rando in range(25):
                            p.stepSimulation()
                            env.move_husky(0,0,0,0)
                        print("===================")
                        print(angle_rel)
                        print("===================")
                        break
                break

            elif abs(angle_rel) <= 40:
                for x in range(50):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            elif abs(angle_rel) <= 90:
                for x in range(80):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            else:
                for x in range(180):
                    p.stepSimulation()
                    env.move_husky(-14, 7, -14, 7)
                for rando in range(40):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

        if angle_rel<0:
            if abs(angle_rel) <= 15:
                while True:
                    img = env.camera_feed()
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                    if(corners[0][0].all()):
                        center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                        center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4

                        y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                        x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                    else:
                        for x in range(2):
                            p.stepSimulation()
                            env.move_husky(1,-1,1,-1)
                            img = env.camera_feed()
                            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
                    
                    y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
                    x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X
                    vector_1 = [x_vector_bot, y_vector_bot]
                
                    vector_2 = [x_vector_rel, y_vector_rel]
                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    dot_product = np.dot(unit_vector_2, unit_vector_1)
                    angle_rel = (np.arccos(dot_product))*(180/np.pi) 

                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)

                    if abs(angle_rel)<=2:
                        for rando in range(25):
                            p.stepSimulation()
                            env.move_husky(0,0,0,0)
                        print("===================")
                        print(angle_rel)
                        print("===================")
                        break
                break

            elif abs(angle_rel) <= 40:
                for x in range(50):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)


            elif abs(angle_rel) <= 90:
                for x in range(80):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(25):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

            else:
                for x in range(180):
                    p.stepSimulation()
                    env.move_husky(7, -14, 7, 14)
                for rando in range(40):
                        p.stepSimulation()
                        env.move_husky(0,0,0,0)

    

    while True:
        while True:
            img = env.camera_feed()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            if(corners[0][0].all()):
                center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
                y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]
                break 
            else:
                for x in range(2):
                    p.stepSimulation()
                    env.move_husky(1,-1,1,-1)

        y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
        x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X


        count =0
        while True:
            count  = count+1
            if(count%50==0):
                print(count)
            if(count>=250):
                break
            if(count%5==0):


                img = env.camera_feed()
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
            
                if(corners[0][0].all()):
                    center_husky_X = (corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
                    center_husky_Y = (corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4

                    y_vector_bot = corners[0][0][0][1] - corners[0][0][3][1]
                    x_vector_bot = corners[0][0][0][0] - corners[0][0][3][0]

                else:
                    for x in range(2):
                        p.stepSimulation()
                        env.move_husky(1,-1,1,-1)
                        img = env.camera_feed()
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        agcorners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)


            p.stepSimulation()
            env.move_husky(10, 10, 10, 10)
            y_vector_rel = blocks[int(target_tile) ]["exact_y-coordinate"] - center_husky_Y
            x_vector_rel = blocks[int(target_tile) ]["exact_x-coordinate"] - center_husky_X

            if abs(y_vector_rel)<=10 and abs(x_vector_rel)<=10:
                for rando in range(50):
                    p.stepSimulation()
                    env.move_husky(0,0,0,0)
                print("-----------------------------am I close enough??-----------------")
                break
        break




cv.waitKey(0)
cv.destroyAllWindows()
