import time
import random
import numpy as np
import math
import TSP_2MST
import os
import sys
import argparse

bestopt = math.inf
node = []
adjM = []
startTime = 0


def readData1(input_file):
    f = open(input_file)

    nodeID = []
    xValue = []
    yValue = []
    i = 1
    for line in f:
        if str(line.split()[0]) == '1':
            nodeID.append(int(line.split()[0]))
            xValue.append(float(line.split()[1]))
            yValue.append(float(line.split()[2]))
            break

    for line in f:
        if line.split()[0] == 'EOF':
            break
        nodeID.append(int(line.split()[0]))
        xValue.append(float(line.split()[1]))
        yValue.append(float(line.split()[2]))
        i = i + 1
    return nodeID, xValue, yValue


def euclidDist(x1, y1, x2, y2):
    return int(round(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))))


def buildMatrix(node, x, y):
    nodeNum = len(node)
    distanceMatrix = np.zeros((nodeNum, nodeNum))
    for i in range(0, len(node)):
        for j in range(i, len(node)):
            distanceMatrix[i, j] = euclidDist(x[i], y[i], x[j], y[j])
            distanceMatrix[j, i] = distanceMatrix[i, j]
    for i in range(0, len(node)):
        distanceMatrix[i, i] = np.NaN
    return distanceMatrix


def reduceMatrix(M, lowerbound):
    lowerbound = lowerbound + np.nansum(np.nanmin(M, axis=1))
    M = (M.transpose() - np.nanmin(M, axis=1)).transpose()
    lowerbound = lowerbound + np.nansum(np.nanmin(M, axis=0))
    M = M - np.nanmin(M, axis=0)
    return M, lowerbound


def branchb(M, lowerbound, tour, cut_off_time):
    global bestopt
    global node
    global adjM
    global startTime
    global outfile2
    # find a tour
    # if M is all NaN, that means the tour is end
    if time.time() - startTime > cut_off_time:
        print("terminated")
        return lowerbound, tour
    if (np.isnan(M)).all():
        if lowerbound < bestopt and len(tour) == len(node) + 1:
            # need to traverse the tour and compute the bestopt here!!
            '''
            lowerbound = 0
            for i in range(0,len(node)):
                lowerbound = lowerbound + adjM[tour[i]][tour[i+1]]
            '''
            bestopt = lowerbound
            outfile2.write(str(time.time() - startTime) + ", " + str(bestopt) + "\n")
            # print('Find a solution with cost:', bestopt)
            # print(tour)
            return lowerbound, tour
        else:
            # not a tour, unable to expand
            return math.inf, []
    # pick arc i, j
    source = tour[-1]
    target = -1
    if len(tour) == len(node):
        target = tour[0]
    else:
        for i in range(0, len(node)):
            if M[source][i] >= 0 and i not in tour:
                target = i
                break
    cycle = False
    if target == -1:
        return math.inf, []
    else:
        # detect cycle
        for i in range(0, len(tour)):
            if tour[i] == target:
                if len(tour) != len(node):
                    # cycle happens
                    cycle = True
                    # duplicate the data
    newM = np.copy(M)
    newtour = tour[:]
    newlowbound = lowerbound
    # arc source, target is in the tour
    if not cycle:
        lowerbound = lowerbound + M[source][target]
        for i in range(0, len(node)):
            M[source][i] = np.NaN
            M[i][target] = np.NaN
        M[target][source] = np.NaN
        M, lowerbound = reduceMatrix(M, lowerbound)
        if lowerbound < bestopt:
            tour.append(target)
            # print('***Left Branch***, lb=',lowerbound)
            # print(tour)
            lowleft, lefttour = branchb(M, lowerbound, tour, cut_off_time)
        else:
            # print('***Left Prune***,lb=',lowerbound)
            lowleft = math.inf
            lefttour = []

    # arc i, j not in tour
    newM[source][target] = np.NaN
    newM, newlowbound = reduceMatrix(newM, newlowbound)
    if newlowbound < bestopt:
        # print('***Right Branch***, lb=',newlowbound)
        # print(newtour)
        lowright, righttour = branchb(newM, newlowbound, newtour, cut_off_time)
    else:
        # print('***Right Prune***, lb=',newlowbound)
        lowright = math.inf
        righttour = []

    if lowleft > lowright and len(righttour) == (len(node) + 1):
        return lowright, righttour
    else:
        if len(lefttour) == (len(node) + 1):
            return lowleft, lefttour
        else:
            return math.inf, []


def bnb_exec(inputfile, cut_off_time):
    global node
    global x
    global y
    global adjM
    global startTime
    global outfile2
    cut_time = cut_off_time
    # input_file="DATA/Cincinnati.tsp"
    input_file =inputfile
    node, x, y = readData1(input_file)
    node[:] = [i - 1 for i in node]
    M = buildMatrix(node, x, y)
    adjM = np.copy(M)
    lowbound = 0
    M, lowbound = reduceMatrix(M, lowbound)
    tour = []
    tour.append(node[0])
    a, b = input_file.split(".")
    city_name = a.split("/")
    city = city_name[-1]
    outfile2 = open("Output/" + city +"_BnB_"+ str(cut_time) + ".trace", 'w')
    startTime = time.time()
    lowbound, tour = branchb(M, lowbound, tour, cut_time)
    endTime = time.time()
    elapsed = endTime - startTime
    print("------result---------")
    print("time:", elapsed)
    print(tour)
    print(lowbound)
    outfile = open("Output/" + city +"_BnB_"+ str(cut_time)+ ".sol", 'w')
    outfile.write(str(lowbound) + "\n")
    for i in range(len(tour) - 1):
        outfile.write((str(tour[i]) + ", " + str(tour[i + 1]) + ", " + str(int(adjM[tour[i]][tour[i + 1]])) + "\n"))
    #outfile2 = open("Output/" + output_file + ".trace", 'w')
    #outfile2.write(str(elapsed) + ", " + str(lowbound))


def readData(input_file):
    f = open(input_file)

    nodeID = []
    xValue = []
    yValue = []
    locs = []
    nodes = []
    numN = 1
    for line in f:
        if str(line.split()[0]) == '1':
            nodeID = int(line.split()[0])
            xValue = float(line.split()[1])
            yValue = float(line.split()[2])
            nodes = [nodeID, xValue, yValue]
            locs.append(nodes)
            break

    for line in f:
        if line.split()[0] == 'EOF':
            break
        nodeID = int(line.split()[0])
        xValue = float(line.split()[1])
        yValue = float(line.split()[2])
        nodes = [nodeID, xValue, yValue]
        locs.append(nodes)
        numN = numN + 1

    nodes = [numN + 1, locs[0][1], locs[0][2]]
    locs.append(nodes)
    # print numN
    return locs, numN


def distance(nodeI, nodeJ):
    dis = math.hypot(nodeI[1] - nodeJ[1], nodeI[2] - nodeJ[2])
    # print nodeI
    # print nodeJ
    # print dis
    return dis


def swap(locs, numN, seedNum):
    newlocs = []
    for i in range(0, numN - seedNum):
        newlocs.append(locs[i + seedNum])
    for i in range(numN - seedNum, numN):
        newlocs.append(locs[i - (numN - seedNum)])
    newlocs.append(newlocs[0])
    return newlocs


def twoOpt(locs, numN, seedNum, cutoff):
    bestDis = 0
    for i in range(0, numN):
        bestDis = bestDis + distance(locs[i], locs[i + 1])

    trace = []
    traceSeed = []
    bestOrder = []

    startTime = time.clock()

    random.seed(seedNum)   
    seedNew=random.randint(0, numN-1)  
    locs = swap(locs, numN, seedNew)
   
    totalDis = 0
    # totalDis=4925582
    order = []

    for nodes in locs:
        order.append(nodes[0])
    for i in range(0, numN):
        totalDis = totalDis + distance(locs[i], locs[i + 1])
    save = 2

    # print order

    while save > 1:
        for i in range(0, numN - 2):
            bef1 = distance(locs[i], locs[i + 1])
            for j in range(i + 2, numN):
                bef2 = distance(locs[j], locs[j + 1])
                aft1 = distance(locs[i], locs[j])
                aft2 = distance(locs[i + 1], locs[j + 1])

                save = bef1 + bef2 - aft1 - aft2
                # print save


                if save > 1:
                    # print save,bef1,bef2,aft1,aft2
                    totalDis = totalDis - save
                    path = order[i + 1:j + 1]
                    rem = locs[i + 1:j + 1]
                    #					print path
                    path.reverse()
                    rem.reverse()
                    #					print path
                    order[i + 1:j + 1] = path
                    locs[i + 1:j + 1] = rem
                    traceSeed.append([totalDis, time.clock() - startTime])
                    # neworder=order
                    break
                    # print distance
                    # print totalDis

            if save > 1:
                break

        if time.clock() - startTime > cutoff:
            break
    trace.append([totalDis, time.clock() - startTime])

    # for i in range (0,)

    # print totalDis
    if totalDis < bestDis:
        bestDis = totalDis
        bestOrder = order

    # print bestDis
    # print trace
    return bestDis, bestOrder, trace, traceSeed


def threeOpt(locs, numN, seedNum, cutoff):
    bestDis = 0
    for i in range(0, numN):
        bestDis = bestDis + distance(locs[i], locs[i + 1])
    trace = []

    bestOrder = []
    traceSeed = []

    random.seed(seedNum)   
    seedNew=random.randint(0, numN-1)  
    locs = swap(locs, numN, seedNew)
    totalDis = 0
    startTime = time.clock()
    # totalDis=4925582
    order = []

    for nodes in locs:
        order.append(nodes[0])
    for i in range(0, numN):
        totalDis = totalDis + distance(locs[i], locs[i + 1])
    save = 2

    # print order

    while save > 1:
        for i in range(0, numN - 4):
            for j in range(i + 2, numN - 2):
                for k in range(j + 2, numN):
                    # print i,j,k
                    ii1 = distance(locs[i], locs[i + 1])
                    jj1 = distance(locs[j], locs[j + 1])
                    kk1 = distance(locs[k], locs[k + 1])
                    ori = ii1 + jj1 + kk1

                    ij = distance(locs[i], locs[j])
                    ij1 = distance(locs[i], locs[j + 1])
                    ik = distance(locs[i], locs[k])

                    i1j1 = distance(locs[i + 1], locs[j + 1])
                    i1k = distance(locs[i + 1], locs[k])
                    i1k1 = distance(locs[i + 1], locs[k + 1])

                    jk = distance(locs[j], locs[k])
                    jk1 = distance(locs[j], locs[k + 1])

                    j1k1 = distance(locs[j + 1], locs[k + 1])

                    # 2opt
                    case21 = ori - (ij + i1j1 + kk1)
                    case22 = ori - (ik + i1k1 + jj1)
                    case23 = ori - (ii1 + jk + j1k1)

                    # 3opt
                    case31 = ori - (ij + i1k + j1k1)
                    case32 = ori - (ij1 + i1k + jk1)
                    case33 = ori - (ik + i1j1 + jk1)
                    case34 = ori - (ij1 + i1k1 + jk)

                    save = max(case21, case22, case23, case31, case32, case33, case34)
                    # save=max(case21,case22,case23,case31,case32,case33,case34)
                    # save=max(case21,case22,case23)
                    # print save
                    if save > 1:
                        if save == case21:
                            totalDis = totalDis - save
                            path = order[i + 1:j + 1]
                            rem = locs[i + 1:j + 1]
                            #					print path
                            path.reverse()
                            rem.reverse()
                            #					print path
                            order[i + 1:j + 1] = path
                            locs[i + 1:j + 1] = rem
                            # neworder=order
                            # print 'case21',totalDis
                            traceSeed.append([totalDis, time.clock() - startTime])
                            break


                        elif save == case22:
                            totalDis = totalDis - save
                            path = order[i + 1:k + 1]
                            rem = locs[i + 1:k + 1]
                            #					print path
                            path.reverse()
                            rem.reverse()
                            #					print path
                            order[i + 1:k + 1] = path
                            locs[i + 1:k + 1] = rem
                            # neworder=order
                            # print 'case22',totalDis
                            traceSeed.append([totalDis, time.clock() - startTime])
                            break


                        elif save == case23:
                            totalDis = totalDis - save
                            path = order[j + 1:k + 1]
                            rem = locs[j + 1:k + 1]
                            #					print path
                            path.reverse()
                            rem.reverse()
                            #					print path
                            order[j + 1:k + 1] = path
                            locs[j + 1:k + 1] = rem
                            # neworder=order
                            # print 'case23',totalDis
                            traceSeed.append([totalDis, time.clock() - startTime])
                            break

                        elif save == case31:
                            totalDis = totalDis - save
                            path = order[i + 1:j + 1]
                            rem = locs[i + 1:j + 1]
                            #					print path
                            path.reverse()
                            rem.reverse()
                            #					print path
                            order[i + 1:j + 1] = path
                            locs[i + 1:j + 1] = rem
                            # neworder=order

                            path = order[j + 1:k + 1]
                            rem = locs[j + 1:k + 1]
                            #					print path
                            path.reverse()
                            rem.reverse()
                            #					print path
                            order[j + 1:k + 1] = path
                            locs[j + 1:k + 1] = rem
                            # print 'case31',totalDis
                            traceSeed.append([totalDis, time.clock() - startTime])
                            break

                        elif save == case32:
                            # print 'case32',order
                            totalDis = totalDis - save
                            temp1 = order[i + 1:j + 1]
                            temp2 = locs[i + 1:j + 1]

                            temp3 = order[j + 1:k + 1]
                            temp4 = locs[j + 1:k + 1]

                            temp3.extend(temp1)
                            temp4.extend(temp2)

                            order[i + 1:k + 1] = temp3
                            locs[i + 1:k + 1] = temp4

                            # print 'case32',totalDis
                            traceSeed.append([totalDis, time.clock() - startTime])
                            break


                        elif save == case33:
                            totalDis = totalDis - save

                            temp1 = order[i + 1:j + 1]
                            temp2 = locs[i + 1:j + 1]

                            temp3 = order[j + 1:k + 1]
                            temp4 = locs[j + 1:k + 1]

                            temp3.reverse()
                            temp4.reverse()

                            temp3.extend(temp1)
                            temp4.extend(temp2)

                            order[i + 1:k + 1] = temp3
                            locs[i + 1:k + 1] = temp4

                            # path=order[i+1:j+1]
                            # rem=locs[i+1:j+1]

                            # path.reverse()
                            # rem.reverse()

                            # order[i+1:j+1]=path
                            # locs[i+1:j+1]=rem
                            traceSeed.append([totalDis, time.clock() - startTime])
                            # print 'case33',totalDis
                            break

                        elif save == case34:
                            totalDis = totalDis - save

                            temp1 = order[i + 1:j + 1]
                            temp2 = locs[i + 1:j + 1]

                            temp3 = order[j + 1:k + 1]
                            temp4 = locs[j + 1:k + 1]

                            temp1.reverse()
                            temp2.reverse()

                            temp3.extend(temp1)
                            temp4.extend(temp2)

                            order[i + 1:k + 1] = temp3
                            locs[i + 1:k + 1] = temp4

                            traceSeed.append([totalDis, time.clock() - startTime])
                            # print 'case34',totalDis
                            break

                if save > 1 or time.clock() - startTime > cutoff:
                    break

            if save > 1 or time.clock() - startTime > cutoff:
                break



                # for i in range (0,)
    trace.append([totalDis, time.clock() - startTime])
    # print totalDis
    if totalDis < bestDis:
        bestDis = totalDis
        bestOrder = order

    # print bestDis
    # print trace
    return bestDis, bestOrder, trace, traceSeed


def LS1(filename, cutoff, randSeed):
    input_file = filename

    locs, numN = readData(input_file)
    seedNum = randSeed

    totalDis, bestOrder, trace, traceSeed = twoOpt(locs, numN, seedNum, cutoff)
    a, b = input_file.split(".")
    city_name = a.split("/")
    city = city_name[-1]

    print
    'totalDis', totalDis
    # print 'best order', bestOrder
    output_file_1 = "Output/%s_LS1_%d_%d.sol" % (city, cutoff, randSeed)
    output = open(output_file_1, 'w')
    output.write(str(int(totalDis)) + "\n")
    for j in range(0, numN):
        output.write(str(bestOrder[j]) + ',' + str(bestOrder[j + 1]) + "\n")

    output_file_3 = "Output/%s_LS1_%d_%d.trace" % (city, cutoff, randSeed)
    output = open(output_file_3, 'w')
    for j in range(0, len(traceSeed)):
        output.write(str("%.2f" % traceSeed[j][0]) + ',' + str("%.4f" % (traceSeed[j][1])) + "\n")


def LS2(filename, cutoff, randSeed):
    input_file = filename

    locs, numN = readData(input_file)
    seedNum = randSeed
    a, b = input_file.split(".")
    city_name = a.split("/")
    city = city_name[-1]


    totalDis, bestOrder, trace, traceSeed = threeOpt(locs, numN, seedNum, cutoff)

    print
    'totalDis', totalDis
    # print 'best order', bestOrder
    output_file_1 = "Output/%s_LS2_%d_%d.sol" % (city, cutoff, randSeed)
    output = open(output_file_1, 'w')
    output.write(str(int(totalDis)) + "\n")
    for j in range(0, numN):
        output.write(str(bestOrder[j]) + ',' + str(bestOrder[j + 1]) + "\n")

    output_file_3 = "Output/%s_LS2_%d_%d.trace" % (city, cutoff, randSeed)
    output = open(output_file_3, 'w')
    for j in range(0, len(traceSeed)):
        output.write(str("%.2f" % traceSeed[j][0]) + ',' + str("%.4f" % (traceSeed[j][1])) + "\n")


def NearestNeighbor(locs, numN, randomnum):
    totalDis = 0
    order = []

    order.append(locs[randomnum])
    last = locs[randomnum]
    while len(order) <= numN:
        mindis = 999999999;
        for nodes in locs[0:numN]:
            if nodes not in order:
                dis = distance(last, nodes)
                if dis < mindis:
                    mindis = dis
                    nextorder = nodes
        if last != nextorder:
            totalDis = totalDis + mindis
            order.append(nextorder)
            last = nextorder
        else:
            break

    totalDis = totalDis + distance(last, locs[randomnum])
    order.append(locs[randomnum])
    return order, totalDis


def approx2(filename, cutoff, randSeed):
    input_file = filename
    locs, numN = readData(input_file)
    a, b = input_file.split(".")
    city_name = a.split("/")
    city = city_name[-1]
    output_file = "Output/" + city + "_ApprxNearestNeighbor_" + str(cutoff) + "_" + str(randSeed) + ".sol"
    outtrace_file = "Output/" + city + "_ApprxNearestNeighbor_" + str(cutoff) + "_" + str(randSeed) + ".trace"

    random.seed(randSeed)

    mindd = 999999999999
    trace = []
    start_DP = time.time()

    while time.time() - start_DP < cutoff:
        randomnum = random.randint(0, numN - 1)
        order, totalDis = NearestNeighbor(locs, numN, randomnum)
        if totalDis < mindd:
            mindd = totalDis
            minorder = order
            trace.append([(time.time() - start_DP) * 1000, mindd])

    if not os.path.isdir("./Output"):
        os.makedirs("Output")
    # ----output .sol -----#
    f = open(output_file, "w")
    finaldistance = str(round(mindd)) + '\n'
    f.write(finaldistance)
    for i in range(0, numN):
        thisline = str(minorder[i][0]) + " " + str(minorder[i + 1][0]) + " " + str(
            round(distance(minorder[i], minorder[i + 1]))) + '\n'
        f.write(thisline)
    f.close()
    # ---------------------#

    # ----output .trace -----#
    f2 = open(outtrace_file, "w")
    for j in range(0, len(trace)):
        f2.write(str("%.2f" % trace[j][0]) + 'ms,' + str(int(trace[j][1])) + "\n")
    f.close()
    # ---------------------#


def main():
    parser = argparse.ArgumentParser(description='Example with non-optional arguments')
    parser.add_argument('-inst', action="store", dest='input_file', type=str)
    parser.add_argument('-alg', action="store", dest='method')
    parser.add_argument('-time', action="store",dest='cut_time', type=int)
    parser.add_argument('-seed', action="store",dest='seed')
    results = parser.parse_args()

    input_file = results.input_file
    method = results.method
    cut_time = results.cut_time
    seed = results.seed
    if not os.path.isdir("./Output"):
        os.makedirs("Output")
    if method == "BnB":
        print("branch and bound")
        cut_time = int(cut_time)
        bnb_exec(input_file, cut_time)
    elif method == "LS1":
        LS1(input_file, int(cut_time), int(seed))
    elif method == "LS2":
        LS2(input_file, int(cut_time), int(seed))
    elif method == "Heur":
        approx2(input_file, int(cut_time), int(seed))
    elif method == "MSTApprox":
        TSP_2MST.main_work(input_file, input_file,int(cut_time), int(seed))


main()