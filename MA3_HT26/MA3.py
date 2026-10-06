""" MA3.py

Student: Maria Gil Sancho 
Mail: maria.gil-sancho.1969@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import functools
import numpy 
from numba import njit
# Exc1
def approximate_pi(n):
    inside_circle=0
    x_axis_inside=[]
    y_axis_inside=[]
    x_axis_outside=[]
    y_axis_outside=[]
    for i in range(n):
         x=random.uniform(-1,1)
         y=random.uniform(-1,1)
         distance=(x**2)+(y**2)
         if distance <= 1:
              inside_circle+=1
              x_axis_inside.append(x)
              y_axis_inside.append(y)
         else:
             x_axis_outside.append(x)
             y_axis_outside.append(y)
     
    pi=4*(inside_circle/n)
    plt.scatter(x_axis_inside,y_axis_inside,color="red")
    plt.scatter(x_axis_outside,y_axis_outside,color="blue")
    plt.savefig(f"Dotplot_for_{n}_points.png")
    return f"Approximation of pi is: {pi} in {n} number of points"

# Exc2, approximation
def sphere_volume(n, d): 
    r=1
    points_of_dimension=[[random.uniform(-1,1) for i in range(d)] for j in range(n)]
    distance_points_of_dimension=[functools.reduce(lambda x,y : x+y, map(lambda x: x**2,x))for x in points_of_dimension]
    filter_inside=list(filter(lambda x: x<=r,distance_points_of_dimension))
    volume_d_hypersphere=2**d*(len(filter_inside)/n)
    return volume_d_hypersphere
    # n is the number of points
    # d is the number of dimensions of the sphere 
   

#Exc2, real value
def hypersphere_exact(n, d):
    r=1
    return numpy.pi**(d/2)/(m.gamma((d/2)+1))*r**d
    # n is the number of points
    # d is the number of dimensions of the sphere 
    

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    r=1
    inside_sphere=0
    for i in range(n):
         distance=0
         for j in range(d):
              x=random.uniform(-1,1)
              distance+=x*x
         if distance<=r:
            inside_sphere+=1
    volume_sphere=2**d*(inside_sphere/n)
    return volume_sphere
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel2(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    return 
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        print(approximate_pi(n))

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}, and approximation = {sphere_volume(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}, and approximation = {sphere_volume(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n} using conventional Python: {stop-start} and volume is {sphere_volume(n,d)}")
    #1. time = 4.1465, volume = 1.878
    #2. time = 3.8202, volume = 1.9619
    #3. time = 3.5755, volume = 1.86368
    print(f"Exc3: Exact volume of {d} and {n} is {hypersphere_exact(n,d)}")
    #1.88
    #Exact volume 1.88, so very prcise.

    print("What is numba time?")
    start_numba=pc()
    sphere_volume_numba(n,d)
    stop_numba=pc()
    print(f"Exc3: Sequential time of {d} and {n} using JIT-compiled version: {stop_numba-start_numba} and volume is {sphere_volume(n,d)}")
    #1. time = 0.6909, volume = 1.933
    #2. time = 0.6368, volume = 2.035
    #3. time = 0.6343, volume = 1.977
    
    #Much faster using Numba library but volume seems more accurate using conventional Python.
          

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")
    
    

if __name__ == '__main__':
	main()
