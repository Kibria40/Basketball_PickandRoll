from shapely.geometry import LineString, box
import numpy as np 


class PlayerTimer:
    def __init__(self):
        self.reset()
    def reset(self):
        self.previous_ball_location = None
        self.current_ball_location = None
        self.current_player = None
        self.previous_player = None
        self.total_proximity_fames = {}
        self.pick_n_roll = False
    def __call__(self, results):
        check_basket = False
        if results.xyxy.size>0:
            balls = results[results.class_id==0]
            basket = results[results.class_id==1]
            players = results[results.class_id==3]
            if basket.xyxy.size>0:
                # print(basket.xyxy)
                basket = basket.xyxy[0]
                check_basket = self.is_basket(basket)
            if balls.xyxy.size>0:
                self.previous_ball_location = self.current_ball_location
                ball_coords = balls.xyxy[0]
                ball_position = np.array([(ball_coords[0] + ball_coords[2])/2, (ball_coords[1] + ball_coords[3])/2],  dtype=np.float32)
                self.current_ball_location = ball_position
                if players.xyxy.size>0:
                    # Calculate the center of each player's bounding box
                    player_centers = np.column_stack([(players.xyxy[:, 0] + players.xyxy[:, 2]) / 2,
                                                    (players.xyxy[:, 1] + players.xyxy[:, 3]) / 2])

                    # Calculate the Euclidean distance between the ball and each player
                    distances = np.linalg.norm(player_centers - ball_position, axis=1)

                    # Find the indices of the two players with the smallest distances
                    closest_player_indices = np.argsort(distances)[0]
                    if players.tracker_id is not None:
                    # Get the bounding boxes and distances of the closest and second closest players
                        player = players.tracker_id[closest_player_indices]
                        self.previous_player = self.current_player
                        self.current_player = player
                        if self.previous_player is not None:
                            if self.previous_player!=self.current_player:
                                self.pick_n_roll = True
                                if player not in self.total_proximity_fames:
                                    self.total_proximity_fames[int(player)] = 1
                                else:
                                    self.total_proximity_fames[int(player)] +=1
                            # else:
                            #     self.pick_n_roll = False
                        
        return self.total_proximity_fames, check_basket
    def is_basket(self, basket_bbox):
        if self.previous_ball_location is not None:
            trajectory_line = LineString([self.previous_ball_location, self.current_ball_location])
            upper_basket_line = LineString([(basket_bbox[0], basket_bbox[1]), (basket_bbox[2], basket_bbox[1])])
            lower_basket_line = LineString([(basket_bbox[0], basket_bbox[3]), (basket_bbox[2], basket_bbox[3])])
            if trajectory_line.intersects(upper_basket_line) and trajectory_line.intersects(lower_basket_line):
                return True
            else:
                return False
        else:
            return False

