#Location coordinates starts at 0. lower left is (0,0), upper right is (w-1,h-1).
#t1 grid from first MDP lecture
w = 4
h = 3
L = 1,1,0;3,2,1;3,1,-1
p = 0.8
r = -0.04

L = [(1,1,0),(3,2,1),(3,1,-1)]


#t2 grid from first MDP lecture
w = 4
h = 3
L = 1,1,0;3,2,1;3,1,-1
p = 0.8
r = 0.04


#t3 grid from first MDP lecture
w = 4
h = 3
L = 1,1,0;3,2,1;3,1,-1
p = 0.8
r = -1

#t4 the cliff from last RL lecture

w = 12
h = 4
L = 1,0,-100;2,0,-100;3,0,-100;4,0,-100;5,0,-100;6,0,-100;7,0,-100;8,0,-100;9,0,-100;10,0,-100;11,0,0
p = 1
r = -1

#t5 the cliff2

w = 12
h = 6
L = 1,0,-100;2,0,-100;3,0,-100;4,0,-100;5,0,-100;6,0,-100;7,0,-100;8,0,-100;9,0,-100;10,0,-100;11,0,0
p = 0.9
r = -1

#t6

w = 5
h = 5
L = 4,0,-10;0,4,-10;1,1,1;3,3,2
p = 0.9
r = -0.5

#t7

w = 5
h = 5
L = 2,2,-2;4,4,-1;1,1,1;3,3,2
p = 0.9
r = -0.25

#t8

w = 7
h = 7
L = 1,1,-4;1,5,-6;5,1,1;5,5,4
p = 0.8
r = -0.5

#t9

w = 7
h = 7
L = 1,1,-4;1,5,-6;5,1,1;5,5,4
p = 0.8
r = -0.5

#t10

w = 7
h = 7
L = 3,1,0;3,5,0;1,1,-4;1,5,-6;5,1,1;5,5,4
p = 0.8
r = -0.25