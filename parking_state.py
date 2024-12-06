class parking_state:
    def __init__(self, debug=False):
        self.debug = debug

        self.total_spots = {}
        self.total_spots['normal'] = 51
        self.total_spots['compact'] = 8
        self.total_spots['electric_vehicle'] = 6
        self.total_spots['RV'] = 4
        self.total_spots['motorcycle'] = 8
        self.total_spots['emergency'] = 1

        self.occupied_spots = {}
        self.occupied_spots['normal'] = self.total_spots['normal']/2
        self.occupied_spots['compact'] = self.total_spots['compact']/2
        self.occupied_spots['electric_vehicle'] = self.total_spots['electric_vehicle']/2
        self.occupied_spots['RV'] = self.total_spots['RV']/2
        self.occupied_spots['motorcycle'] = self.total_spots['motorcycle']/2
        self.occupied_spots['emergency'] = self.total_spots['emergency']/2

    def _incr_spot(self, type):
        self.occupied_spots[type] = self.occupied_spots[type] + 1

    def _decr_spot(self, type):
        self.occupied_spots[type] = self.occupied_spots[type] - 1

    def update(self, entrance_vehicle_class, exit_vehicle_class):
        changed = 0
        if entrance_vehicle_class:
            self._incr_spot('normal')
            changed = changed + 1
            if self.debug:
                print(f'State: incremented spot type normal for entering vehicle {entrance_vehicle_class}')
        
        if exit_vehicle_class:
            self._decr_spot('normal')
            changed = changed + 1
            if self.debug:
                print(f'State: incremented spot type normal for exiting vehicle {entrance_vehicle_class}')
        
        return changed

    def print(self):
        print(self.occupied_spots)