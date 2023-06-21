from queue import PriorityQueue

q = PriorityQueue()
q.put((96, (19,0)))
q.put((94, (21,1)))
# q.put((97, (0,1)))
print(q.get())
print(q.get())
# print(q.get())
