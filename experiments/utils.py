
def get_node_values(observables):

	X = ['n0','n1','n2','n3','n4']
	Z = ['n5','n6']
	A = ['n7','n8']

	V = {key:0 for key in X+Z+A}
	
	taxi_row, taxi_col, passenger_pos, goal_pos = observables[0], observables[1], observables[2], observables[3]

	V['n0'] = passenger_pos
	V['n4'] = goal_pos

	V['n2'] = [taxi_row, taxi_col]

	if V['n2'] == [0,0]:
		V['n2'] = 0
	elif V['n2'] == [0,4]:
		V['n2'] = 1
	elif V['n2'] == [4,0]:
		V['n2'] = 2
	elif V['n2'] == [4,3]:
		V['n2'] = 3

	V['n5'] = int(V['n0'] == 4)	
	V['n1'] = int(V['n0'] == V['n2'])
	V['n3'] = int(V['n4'] == V['n2'])
	
	return V

def interventional_selection(s, G, V):

	if(V['n5'] == 1):
		Z = ['n6']
		A = ['n8']
	else:
		Z = ['n5']
		A = ['n7']

	if(A == ['n8']):
		if(V['n3'] == 1):
			return 5
	if(A == ['n7']):
		if(V['n1'] == 1):
			return 4
		
	return None
