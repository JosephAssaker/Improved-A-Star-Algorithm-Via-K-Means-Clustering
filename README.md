# Improved A* Algorithm Via The Use Of K-Means Clustering

<img src="./images/traffic.jpg" style="width:1000px;margin-bottom:15px">

<span>Photo by <a href="https://unsplash.com/@dnevozhai?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Denys Nevozhai</a> on <a href="https://unsplash.com/photos/aerial-photography-of-concrete-roads-7nrsVjvALnA?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a></span>.
  

# Overview

The goal of this project is to explore ways in which **finding the shortest path** between any two nodes in a map could be optimized.
<br/>More precisely, we consider in this work a **traffic congestion map**, meaning that other than the distance factor, the speed factor needs to be also taken into account to ultimately find the *fastest* path between nodes.

The main idea is to use a costly but precise implementation of the A* algorithm that make use of an exact *heuristic*. We argue that even though this step is computationally costly, it can be done *offline* once and updated rarely when map characteristics change.
<br/>On the other hand, traffic is a dynamic parameter that changes very frequently. Thus an efficient and fast handling must be done to take this parameter into account, and we do that in this work via the use of K-means clustering.

# Usage

* **Zero** python module installations are required to run this project. It makes use of only built-in python modules as well as a provided `graphics.py` module.
* Maps in this project are real-world maps that are first downloaded from [Open Street Map](https://www.openstreetmap.org/) and then converted to XML format for better handling in Python via [SUMO](https://sumo.dlr.de/userdoc/index.html).  
  * SUMO can be installed following the instructions [here](https://sumo.dlr.de/userdoc/Installing/index.html) and maps can be converted using the [netconvert](https://sumo.dlr.de/userdoc/netconvert.html) command.  
  * An example map is already provided in `maps/map.net.xml`.

---

## **Table of Content:**

 1. Introduction
 2. Optimizing A* Algorithm Via a Perfect Heuristic Function
 3. Utilizing The K-means Algorithm In Order To Avoid Traffic
 4. Results
 5. Conclusion and Future Work
 6. References
 
Continue reading the whole report [here](report/Improved%20A%20Star%20Algorithm%20via%20the%20use%20of%20K-means%20Clustering%20-%20Report.pdf).
