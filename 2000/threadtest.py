
import queue
import threading
import time

exitFlag = 0

class myThread (threading.Thread):
   def __init__(self, threadID, name, q):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.q = q
   def run(self):
      print ("Starting " + self.name)
      process_data(self.name, self.q)
      print ("Exiting " + self.name)

def process_data(threadName, q):
   while not exitFlag:
      queueLock.acquire()
      if not workQueue.empty():
         data = q.get()
         queueLock.release()
         #print ("%s processing %s" % (threadName, data))
         s = ("%s processing %s" % (threadName, data))
         queueLock.acquire()
         resultQueue.put(s)
         queueLock.release()
      else:
         queueLock.release()
      time.sleep(1)

threadList = ["Thread-1", "Thread-2", "Thread-3","Thread-4", "Thread-5", "Thread-6"]

queueLock = threading.Lock()
workQueue = queue.Queue(0)
resultQueue = queue.Queue(0)
threads = []

# Create new threads
for threadID, tName in enumerate(threadList):
   thread = myThread(threadID+1, tName, workQueue)
   thread.start()
   threads.append(thread)

# Fill the queue
queueLock.acquire()
workList = ["One", "Two", "Three", "Four", "Five"]
for word in workList:
   workQueue.put(word)
queueLock.release()

# Wait for queue to empty
while not workQueue.empty():
   pass

# Notify threads it's time to exit
exitFlag = 1

# Wait for all threads to complete
for t in threads:
   t.join()

# Print results:
while not resultQueue.empty():
   s = resultQueue.get()
   print(s)

print ("Exiting Main Thread")