
class GridSearchCmdGen(object):
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
				params += ' '+str(keys[j])+' '+str(self.parameters[keys[j]][current_state[j]])
			cmds.append(params)

			for j in range(len(current_state)):
				if((current_state[j]+1)%len(self.parameters[keys[j]]) == 0):
					current_state[j] = 0
				else:
					current_state[j] += 1
					break;
		return cmds
