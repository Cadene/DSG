import random
import sys
from threading import Thread, Lock
import time
import os
import StringIO
import subprocess
import re

import signal
#class OARSubCommunicate(Thread):
exit = {'status':False}
exit_m = Lock()
class OARSubStatChecker(object):
	def __init__(self, ended, ended_m, running, running_m):
		self.ended  = ended
		self.ended_m = ended_m
		self.running = running
		self.running_m = running_m
	def run(self):

		out     = os.popen('oarstat').read()
		groups  = re.findall('\n(\d+)',out)
		self.ended_m.acquire()
		self.running_m.acquire()
		for rp in self.running:
			if(not(rp in groups)):
				#Process is finished
				self.ended.append(rp)
				self.running.remove(rp)

		self.ended_m.release()
		self.running_m.release()



class OARSubIO(Thread):
	def __init__(self, queue, queue_m, running, running_m):
		Thread.__init__(self)
		self.queue_m = queue_m
		self.queue   = queue
		self.running = running
		self.running_m = running_m

	def kill_all_running_process(self):
		self.running_m.acquire()
		for pid in self.running:
			os.popen('oardel '+str(pid))
			print('Deleting jjob with Id = '+str(pid))
		self.running_m.release()

	def command(self, cmd):
		if(cmd == 'queue'):

			self.queue_m.acquire()
			if(len(self.queue) == 0):
				print('All job finished')
			else:
				for i in range(len(self.queue)):
					print(self.queue[i])
			self.queue_m.release()
		if(cmd == 'n_running'):
			self.running_m.acquire()
			print(str(len(self.running))+'')
			self.running_m.release()
		if(cmd == 'n_queue'):
			self.queue_m.acquire()
			print(str(len(self.queue))+'')
			self.queue_m.release()
		if(cmd == 'exit'):
			exit_m.acquire()
			exit['status'] = True
			exit_m.release()
			self.kill_all_running_process()
			sys.exit()

	def run(self):
		while(True):
			cmd = raw_input("Please type your command $: ")
			self.command(cmd)

class OARSubLauncher(object):
    def __init__(self,binary= 'sh /home/gerald/TempNN/script/gridsearch/CNN.sh', parameters='-t 5'):
        self.parameters = parameters
        self.binary = binary
    def start(self,id):
		#
		out = os.popen('oarsub -l core=1,walltime=0:45:0  "'+self.binary+' '+self.parameters+'"').read()
		m = re.findall("(\d+)", out)
		self.oarpid = m[0]
		return self.oarpid
		#print('OAR process launched with binary : '+self.binary+' and parameters : '+self.parameters)


class OARSubPersonalServer(object):
	def __init__(self,max_threads):
		self.max_threads = max_threads
		self.queue   = []
		self.ended   = []
		self.running = []
		self.queue_m, self.ended_m, self.running_m = Lock(),Lock(),Lock()
	def add(self,binary,parameters):
		self.queue_m.acquire()
		self.queue.append([binary,parameters])
		self.queue_m.release()
	def start(self):
		self.OARSSC = OARSubStatChecker(self.ended, self.ended_m, self.running, self.running_m)
		self.OARSIO = OARSubIO(self.queue, self.queue_m, self.running, self.running_m)
		self.OARSIO.start()
		self.index = {}
		self.current_id = 0
		while(True):



			os.system('sleep 1')
			self.OARSSC.run()
			#Exit status
			self.queue_m.acquire()
			self.running_m.acquire()
			exit_m.acquire()

			if(exit['status']):
				self.running_m.release()
				self.queue_m.release()
				print('Exiting please wait...')
				exit_m.release()
				oldstdin = sys.stdin
				sys.stdin = StringIO.StringIO('exit\n')
				self.OARSIO.join()
				sys.exit()
			exit_m.release()
			#Launching task

			if(len(self.queue)>0):
				#print(len(self.queue))
				n_running = len(self.running)
				for i in range(min(len(self.queue),self.max_threads - n_running)):
					p = self.queue.pop()
					#print(p)
					launcher = OARSubLauncher(p[0],p[1])
					self.index[launcher.start(self.current_id)] = self.current_id
					self.current_id += 1
					os.system('sleep 1')
					self.running.append(launcher.oarpid)
			self.running_m.release()
			self.queue_m.release()





class OARGridSearch(object):
	def __init__(self,bin):
		self.bin = bin
	def set_parameters(self,parameters):
		self.parameters = parameters
	def compute_cmd(self):
		keys = self.parameters.keys()
		current_state = [0 for key in keys]
		index_key = 0

		#Computing possibilities
		n_cmd = 1
		for key in keys :
			n_cmd *= len(self.parameters[key])
		cmds = []
		for i in range(n_cmd):
			params = ''
			for j in range(len(current_state)):
				params += ' -'+str(keys[j])+' '+str(self.parameters[keys[j]][current_state[j]])
			cmds.append(params)

			for j in range(len(current_state)):
				if((current_state[j]+1)%len(self.parameters[keys[j]]) == 0):
					current_state[j] = 0
				else:
					current_state[j] += 1
					break;
		return cmds
signal.signal(signal.SIGINT, signal.SIG_IGN)



parameters = {}

#folder
parameters['-dirpred'] = ['data/prediction/logistic']
parameters['-dirmodel'] = ['models/logistic']
parameters['-dirlog'] = ['logs/logistic']

#initialiation
parameters['-random_state'] = [2]

#learning parameters
parameters['-C'] = [0.1,1,2,3,4,5,6,7,8,9,10]
parameters['-class_weight'] = ['balanced', None]



GS = OARGridSearch('sh src/launcher/launcher_remi_logistic.sh')
GS.set_parameters(parameters)
cmds = GS.compute_cmd()
serv = OARSubPersonalServer(20)
for cmd in cmds:
	serv.add('sh src/launcher/launcher_remi_logistic.sh',cmd)
serv.start()
