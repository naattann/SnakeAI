import PySimpleGUI as sgui
from random import randint


class GameClass:
	
	def reset(self):

		global  direction, DIRECTIONS, apple_pos, game_iteration_nr, reward
			
		self.score = 0
		reward = 0
		self.game_iteration_nr = 0
		self.snake_body = [(5,4),(4,4),(3,4),(2,4),(1,4)]											
		DIRECTIONS = [(1,0),(0,-1), (-1,0), (0,1)]
		self.direction = (1,0)
		self.apple_pos = self.place_apple()
		self.apple_eaten = False
		self.distanceCheck = True
	
	def __init__(self):

		self.dang1 = False
		self.dang2 = False
		self.dang3 = False
		self.dang4 = False

		self.body_segments = []
		self.distanceCheck = True
		self.CELL_NUM= 8
		self.FIELD_SIZE = self.CELL_NUM * 50
		self.CELL_SIZE = self.FIELD_SIZE/ self.CELL_NUM		
		sgui.theme('Darkbrown3')
		self.field = sgui.Graph(
		canvas_size = (self.FIELD_SIZE,self.FIELD_SIZE),
		graph_bottom_left = (0,0),
		graph_top_right = (self.FIELD_SIZE,self.FIELD_SIZE),
		background_color = 'white')
		self.layout = [[self.field]]		
		self.window = sgui.Window('AutonomousSnake', self.layout,return_keyboard_events = True,finalize=True)				
		self.reset()

	def convert_cell_to_pixel(self,cell):

		upper_left_cell_corner = cell[0] * self.CELL_SIZE, cell[1] * self.CELL_SIZE
		lower_right_cell_corner = upper_left_cell_corner[0] + self.CELL_SIZE, upper_left_cell_corner[1] + self.CELL_SIZE
		return upper_left_cell_corner, lower_right_cell_corner


	def place_apple(self):

		
		self.apple_pos = randint(0,self.CELL_NUM - 1), randint(0,self.CELL_NUM - 1)
		while self.apple_pos in self.snake_body:
			self.apple_pos = randint(0,self.CELL_NUM - 1), randint(0,self.CELL_NUM - 1)
		return self.apple_pos


	def screen_refresh(self):

				#Remove old screen state
				
				self.field.DrawRectangle((0,0),(self.FIELD_SIZE,self.FIELD_SIZE), 'white')        

				#Draw apple

				upper_left_cell_corner, lower_right_cell_corner = self.convert_cell_to_pixel(self.apple_pos)
				self.field.DrawRectangle(upper_left_cell_corner,lower_right_cell_corner,'pink')

				# Draw snake

				for index, part in enumerate(self.snake_body):
					upper_left_cell_corner, lower_right_cell_corner = self.convert_cell_to_pixel(part)
					color = 'red' if index == 0 else 'brown'
					self.field.DrawRectangle(upper_left_cell_corner,lower_right_cell_corner,color)

	
	def collision(self):

				if not 0 <= self.snake_body[0][0] <= self.CELL_NUM - 1 or \
				   not 0 <= self.snake_body[0][1] <= self.CELL_NUM - 1 or \
				   self.snake_body[0] in self.snake_body[1:]:
					
					
					return True
				return False	


	def _move(self,action):

		global reward
		reward = 0


		



		event, values = self.window.read(timeout = 30)

		idx = DIRECTIONS.index(self.direction)
		
		# Checking index of current direction
		
		if action == [0,1,0]:
			self.direction =  self.direction

			#no change

		elif action == [0,0,1]:
			idx = (idx+1) % 4
			self.direction =  DIRECTIONS[idx]      #turn right
			
		else:
			idx = (idx-1) % 4					#turn left
			self.direction =  DIRECTIONS[idx] 
			

				# Eating apple
		if self.snake_body[0] == self.apple_pos:
			self.apple_pos = self.place_apple()
			self.apple_eaten = True			
			self.score += 1
			self.distanceCheck = False
			reward = 20
			
		oldx = self.snake_body[0][0]
		oldy = self.snake_body[0][1]
		applex = self.apple_pos[0]
		appley = self.apple_pos[1]
				# Snake update
		new_head = (self.snake_body[0][0] + self.direction[0],self.snake_body[0][1] + self.direction[1])
		self.snake_body.insert(0,new_head)
		if self.apple_eaten == False:
			self.snake_body.pop()
	


		self.body_segments = self.snake_body[3:-2]
		self.apple_eaten = False
		newx = self.snake_body[0][0]
		newy = self.snake_body[0][1]


		if self.distanceCheck == True:

			if oldx >= applex:

				olddiffx = oldx - applex
				newdiffx = newx - applex

			else:
				olddiffx = applex-oldx
				newdiffx = applex-newx


			if oldy >= appley :

				olddiffy = oldy - appley
				newdiffy = newy - appley

			else:
				olddiffy = appley - oldy
				newdiffy = appley - newy

			olddiff = olddiffx + olddiffy
			newdiff = newdiffx + newdiffy

			if olddiff > newdiff :

				reward = -1

			if newdiff > olddiff :

				reward = 1

			else:

				reward = 0
		else:
			self.distanceCheck = True
	

		
		self.dang1 = self.body_segments[:] == self.snake_body[0] + (0,1) 
		self.dang2 = self.body_segments[:] == self.snake_body[0] + (0,-1) 
		self.dang3 = self.body_segments[:] == self.snake_body[0] + (-1,0) 
		self.dang4 = self.body_segments[:] == self.snake_body[0] + (1,0)
		

		return reward
	
	
	def play_step(self,action):

			global reward
			
			
			game_over = False
			self.game_iteration_nr += 1


			reward= self._move(action)
			
			if self.collision():
				reward = -20
				game_over = True
				
				
				return reward, game_over, self.score

			if self.game_iteration_nr > len(self.snake_body)* 3 + 100:
				reward = -15
				game_over = True
				

			
			self.screen_refresh()

			return reward, game_over, self.score
